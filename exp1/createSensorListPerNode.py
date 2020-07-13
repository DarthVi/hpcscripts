"""
2020-06-23 18:48:10

@author: Vito Vincenzo Covella
""" 

import os
import pathlib
from tqdm import tqdm

"""
check if sensor matches one of the nodes
"""
def checkNodeInList(nodelist, sensor):
    for node in nodelist:
        if node in sensor.replace('/', ''):
            return True
    return False

if __name__ == '__main__':
    here = pathlib.Path(__file__).parent
    sensors = []

    #open the file with the list of nodes we are interested in
    with open(here.joinpath("compute_nodes.txt")) as comp_nodes_file:
        nodes = list(map(str.strip, comp_nodes_file))

    #open the file with the list of all sensors
    with open(here.joinpath("sensorlist_trimmed.txt")) as full_sensor_file:
        full_sensorlist = list(map(str.strip, full_sensor_file))

    print("Collecting sensors matching compute nodes")
    for sensor in tqdm(full_sensorlist):
        #if the sensor matches one of the nodes, add it to the list
        if checkNodeInList(nodes, sensor) == True:
            sensors.append(sensor)

    #save a file with the list of sensors that match the compute nodes
    print("Saving sensors to file")
    with open(here.joinpath("sensorlist_data.txt"), 'w') as f:
        for item in sensors:
            f.write("%s\n" % item)