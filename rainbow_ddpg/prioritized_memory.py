import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from threading import RLock, Condition


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


# Create a memory object. The object is synchronized using locks so we can have
# Rollout worker/workers on a separate thread from training.  The class allows the
# user to synchronize episode collection and training by imposing a limit, which grows
# after each training episode. This was used during development to replicate the ratio
# of number of collected transitions to the number of training steps.
class Memory(object):
    def __init__(self, limit, nb_rollout_steps=100):
        self.lock = RLock()  # Underlying lock.
        self.condition = Condition(self.lock)  # Used to synchronize training and data collection.
        self.storage = []  # Actual underlying data structure.
        self.maxsize = limit  # Maximal size.
        self._next_idx = 0  # Pointer to storage
        self.adding_demonstrations = True  # Whether the initial period of collecting demonstrations is done.
        self.num_demonstrations = 0  # Number of demo transitions in the buffer.
        self.nb_rollout_steps = nb_rollout_steps
        self._total_transitions = 0  # Total number of collected transitions (should grow linearly)
        self.total_transition_limit = self.nb_rollout_steps  # Limit on the rate of collecting new data.
        self.storable_elements = [
            "states0", "obs0", "actions", "rewards", "states1", "obs1",
            "terminals1", "aux0", "aux1",
            "demo_state_id"
        ]  # List of entities we want to store for each timestep.

    def __len__(self):
        with self.lock:
            return len(self.storage)

    @property
    def total_transitions(self):
        with self.lock:
            return self._total_transitions

    def grow_limit(self):
        with self.condition:
            self.total_transition_limit += self.nb_rollout_steps
            self.condition.notify_all()

    @property
    def nb_entries(self):
        with self.lock:
            return len(self.storage)

    def append(self, *args, training=True, count=True):
        # Demonstrations are not counted towards the _total_transitions.
        if count:
            self._total_transitions += 1

        with self.condition:
            while self.total_transitions >= self.total_transition_limit:
                self.condition.wait()

            # Sanity check the size of entries. Note that the ordering is not checked and must
            # match the order defined in self.storable elements.
            assert len(args) == len(self.storable_elements)
            if not training:
                return False
            entry = args
            if self._next_idx >= len(self.storage):
                self.storage.append(entry)
            else:
                self.storage[self._next_idx] = entry

            self._next_idx = int(self._next_idx + 1)
            if self._next_idx >= self.maxsize:
                self._next_idx = self.num_demonstrations
            return True

    def append_demonstration(self, *args, **kwargs):
        with self.lock:
            assert len(args) == len(self.storable_elements)
            assert self.adding_demonstrations
            if not self.append(*args, count=False, **kwargs):
                return
            self.num_demonstrations += 1

    def _get_batches_for_idxes(self, idxes):
        with self.lock:
            # Create a dictionary with a list for each storable entity.
            batches = {
                storable_element: [] for storable_element in self.storable_elements
            }
            for i in idxes:
                entry = self.storage[i]
                # Sanity check the size of entries
                assert len(entry) == len(self.storable_elements)
                for j, data in enumerate(entry):
                    batches[self.storable_elements[j]].append(data)
            result = {k: array_min2d(v) for k, v in batches.items()}
            return result

    def sample(self, batch_size):
        with self.lock:
            idxes = np.random.random_integers(
                low=0, high=self.nb_entries - 1, size=batch_size)
            demos = [i < self.num_demonstrations for i in idxes]
            encoded_sample = self._get_batches_for_idxes(idxes)
            encoded_sample['weights'] = array_min2d(np.ones((batch_size, )))
            encoded_sample['idxes'] = idxes
            encoded_sample['demos'] = array_min2d(demos)
            return encoded_sample

    def demonstrations_done(self):
        with self.lock:
            self.adding_demonstrations = False

    def sample_rollout(self, batch_size, nsteps, _, gamma, pretrain=False):
        with self.lock:
            batches = self.sample(batch_size)
            n_step_batches = {
                storable_element: []
                for storable_element in self.storable_elements
            }
            n_step_batches["step_reached"] = []
            idxes = batches["idxes"]
            for idx in idxes:
                local_idxes = list(range(idx, min(idx + nsteps, len(self))))
                transitions = self._get_batches_for_idxes(local_idxes)
                summed_reward = 0
                count = 0
                terminal = 0.0
                terminals = transitions['terminals1']
                r = transitions['rewards']
                for i in range(len(r)):
                    summed_reward += (gamma**i) * r[i]
                    count = i
                    if terminals[i]:
                        terminal = 1.0
                        break
                n_step_batches["step_reached"].append(count)
                n_step_batches["obs1"].append(transitions["obs1"][count])
                n_step_batches["terminals1"].append(terminal)
                n_step_batches["rewards"].append(summed_reward)
                n_step_batches["states1"].append(transitions["states1"][count])
                n_step_batches["aux1"].append(transitions["aux1"][count])
                n_step_batches["actions"].append(transitions["actions"][0])
            n_step_batches['demos'] = batches['demos']
            n_step_batches = {
                k: array_min2d(v)
                for k, v in n_step_batches.items()
            }
            n_step_batches['weights'] = batches['weights']
            n_step_batches['idxes'] = idxes
            n_step_batches['weights'] = batches['weights']
            return batches, n_step_batches, sum(batches['demos']) / batch_size

    def update_priorities(self, idxes, td_errors, actor_losses=0.0):
        pass


class PrioritizedMemory(Memory):
    def __init__(self,
                 limit,
                 alpha,
                 transition_small_epsilon=1e-6,
                 demo_epsilon=0.2,
                 nb_rollout_steps=100):
        super(PrioritizedMemory, self).__init__(limit, nb_rollout_steps)
        assert alpha > 0
        self._alpha = alpha
        self._transition_small_epsilon = transition_small_epsilon
        self._demo_epsilon = demo_epsilon
        it_capacity = 1
        while it_capacity < self.maxsize:
            it_capacity *= 2  # Size must be power of 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, *args, **kwargs):
        with self.condition:
            idx = self._next_idx
            if not super().append(*args, **kwargs):
                return
            self._it_sum[idx] = self._max_priority
            self._it_min[idx] = self._max_priority

    def append_demonstration(self, *args, **kwargs):
        with self.lock:
            idx = self._next_idx
            if not super().append(*args, **kwargs, count=False):
                return
            self._it_sum[idx] = self._max_priority
            self._it_min[idx] = self._max_priority
            self.num_demonstrations += 1

    def _sample_proportional(self, batch_size, pretrain):
        with self.lock:
            res = []
            if pretrain:
                res = np.random.random_integers(
                    low=0, high=self.nb_entries - 1, size=batch_size)
                return res
            for _ in range(batch_size):
                while True:
                    mass = np.random.uniform(
                        0, self._it_sum.sum(0,
                                            len(self.storage) - 1))
                    idx = self._it_sum.find_prefixsum_idx(mass)
                    if idx not in res:
                        res.append(idx)
                        break
            return res

    def sample_prioritized(self, batch_size, beta, pretrain=False):
        with self.lock:
            idxes = self._sample_proportional(batch_size, pretrain)
            demos = [i < self.num_demonstrations for i in idxes]
            weights = []
            p_sum = self._it_sum.sum()
            for idx in idxes:
                p_sample = self._it_sum[idx] / p_sum
                weight = ((1.0 / p_sample) * (1.0 / len(self.storage)))**beta
                weights.append(weight)
            weights = np.array(weights) / np.max(weights)
            encoded_sample = self._get_batches_for_idxes(idxes)
            encoded_sample['weights'] = array_min2d(weights)
            encoded_sample['idxes'] = idxes
            encoded_sample['demos'] = array_min2d(demos)
            return encoded_sample

    def sample_rollout(self, batch_size, nsteps, beta, gamma, pretrain=False):
        with self.lock:
            batches = self.sample_prioritized(batch_size, beta, pretrain)
            n_step_batches = {
                storable_element: []
                for storable_element in self.storable_elements
            }
            n_step_batches["step_reached"] = []
            idxes = batches["idxes"]
            for idx in idxes:
                local_idxes = list(range(idx, min(idx + nsteps, len(self))))
                transitions = self._get_batches_for_idxes(local_idxes)
                summed_reward = 0
                count = 0
                terminal = 0.0
                terminals = transitions['terminals1']
                r = transitions['rewards']
                for i in range(len(r)):
                    summed_reward += (gamma**i) * r[i]
                    count = i
                    if terminals[i]:
                        terminal = 1.0
                        break
                n_step_batches["step_reached"].append(count)
                n_step_batches["obs1"].append(transitions["obs1"][count])
                n_step_batches["terminals1"].append(terminal)
                n_step_batches["rewards"].append(summed_reward)
                n_step_batches["states1"].append(transitions["states1"][count])
                n_step_batches["aux1"].append(transitions["aux1"][count])
                n_step_batches["actions"].append(transitions["actions"][0])
            n_step_batches['demos'] = batches['demos']
            n_step_batches = {
                k: array_min2d(v)
                for k, v in n_step_batches.items()
            }
            n_step_batches['weights'] = batches['weights']
            n_step_batches['idxes'] = idxes
            n_step_batches['weights'] = batches['weights']
            return batches, n_step_batches, sum(batches['demos']) / batch_size

    def update_priorities(self, idxes, td_errors, actor_losses=0.0):
        with self.lock:
            priorities = td_errors + \
                (actor_losses ** 2) + self._transition_small_epsilon
            for i in range(len(priorities)):
                if idxes[i] < self.num_demonstrations:
                    priorities[i] += np.max(priorities) * self._demo_epsilon
            assert len(idxes) == len(priorities)
            for idx, priority in zip(idxes, priorities):
                assert priority > 0
                assert 0 <= idx < len(self.storage)
                self._it_sum[idx] = priority**self._alpha
                self._it_min[idx] = priority**self._alpha
                self._max_priority = max(self._max_priority, priority**self._alpha)
