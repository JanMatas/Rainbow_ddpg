import xml.etree.ElementTree as ET
import uuid
import numpy as np
import os
import sys


# Randomizes the appearance of the arm and creates a corresponding URDF file. Returns
# the path to the URDF file and also the colors used.
def create_randomized_description():
    rootdir = os.path.dirname(sys.modules['__main__'].__file__)
    tree = ET.parse(rootdir + 'mico_description/urdf/mico.urdf')
    root = tree.getroot()
    links = root.findall(".//material[@name='carbon_fiber']")
    link_color = np.clip(
        np.random.normal([0.1, 0.1, 0.1], 0.05), [0, 0, 0], [1, 1, 1])

    for link in links:
        link[0].set("rgba", "{} {} {} 1".format(*link_color))
    rings = root.findall(".//material[@name='carbon_ring']")
    ring_color = np.clip(
        np.random.normal([0.4, 0.4, 0.4], 0.05), [0, 0, 0], [1, 1, 1])
    for ring in rings:
        ring[0].set("rgba", "{} {} {} 1".format(*ring_color))
    tmp_dir = rootdir + "tmp/"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    fn = tmp_dir + str(uuid.uuid1()) + ".urdf"
    print(fn)
    tree.write(fn)
    return fn, link_color, ring_color


if __name__ == '__main__':
    print(create_randomized_description())
