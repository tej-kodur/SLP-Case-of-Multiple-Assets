""" Square topology with Multiple sources/single source and using Biased coin random walk with Shortest path
(Mostly used for Backward walk)  """

# -----------Importing Required Files----------#

import math
import time
from datetime import datetime
import random
import pandas as pd
from tqdm import tqdm_gui, tqdm

start = time.perf_counter()


class attacker:
    '''Stores Attacker current and past visited locations for future use'''

    def __str__(self):
        return "Attacker at " + str(self.location) + " and visited " + str(
            len(self.visitedLocations)) + " nodes till now."

    def __init__(self):
        self.initial_location = [0, 0]
        self.location = list(self.initial_location)

        visitedLocations = []
        self.visitedLocations = visitedLocations

    def getLocation(self, position):
        self.location = position
        self.visitedLocations.append(position)

    def isVisited(self, positon):
        if positon in self.visitedLocations:
            return True
        else:
            return False

    def isNotMoved(self):
        if self.location == self.initial_location:
            return True
        else:
            return False

    @classmethod
    def getSNLocations(cls, positions):
        cls.snLocations = list(positions)

    def isFoundSN(self):
        if self.location in attacker.snLocations:
            return True
        else:
            return False

    def clearLocations(self):

        self.visitedLocations.clear()

    def isVisited(self, position):
        if position in self.visitedLocations:
            return True
        else:
            return False


class node:
    '''Stores every Node details like current position,energy,status,type,etc., for future use'''

    def __str__(self):
        return str(self.type) + " Node Id " + str(self.id) + " at " + str(self.location) + " with energy " + str(
            self.energy) + " and status is " + str(self.status) + " also sent " + str(
            self.sentpackets) + " packets till now."

    def __init__(self, location):
        self.location = location
        id = len(nodes) + 1
        self.id = id

        initialEnergy = 500000000  # 5000000       # 500000000nJ  = 0.5J
        self.initialEnergy = initialEnergy
        energy = initialEnergy
        self.energy = energy
        status = "Active"
        self.status = status
        type = "Normal Node"
        self.type = type
        sentpackets = 0
        self.sentpackets = sentpackets
        self.knowHopCount = False
        self.uniquePackets = 0

    def getDist(self, distance):
        self.disance = distance

    def getremEnergy(self, remEnergy):
        self.energy = remEnergy

    def getStatus(self):
        self.status = "Inactive"
        self.energy = 0

    def getType(self, name):
        self.type = name

    def packetsent(self):
        self.sentpackets = self.sentpackets + 1

    def getProbability(self, probability):
        self.probability = probability
        halfEntropy = self.probability * math.log(self.probability, 2)
        self.halfEntropy = halfEntropy

    def getHopCount(self, hopCount):
        self.hopCount = hopCount
        self.knowHopCount = True

    @classmethod
    def giveEnergyDetails(cls):
        """ Details for Calculating Energy """

        cls.Eelec = 50  # 50 nJ/bit
        cls.l = 6392  # Data bits per message
        cls.Eamp = 0.0000013  # nJ/bit/m^4
        cls.Efs = 0.01  # nJ/bit/m^2
        cls.d0 = 87  # m

        return cls.Eelec, cls.l, cls.Eamp, cls.Efs, cls.d0

    def getUniquePacket(self):
        self.uniquePackets += 1


nodes = []


##--------Generate nodes Square/Rectangle-------------##

def square(length, breadth, radio_range):
    offset_dist = int((2 ** 0.5 / 2) * radio_range)  # Offset distance formula

    # print("Offset distance = ", offset_dist)

    # GENERATING NODES.

    q_2 = []
    q_3 = []
    q_4 = []
    x_cord = 0  # temporaru stores x ordinate values
    q_1 = []  # Empty list to store Quadrant 1 nodes including positive x and y axis
    for i in range(int((breadth / 2) / offset_dist) + 1):
        if i == 0:  # Without this and line 38, points on X and Y-axis can't be generated
            x_cord = 0
        else:
            x_cord += offset_dist
        y_cord = 0  # Temporary stores y ordinate values
        for j in range(int((length / 2) / offset_dist) + 1):
            if j == 0:
                y_cord = 0
            else:
                y_cord += offset_dist
            q_1.append([int(x_cord), int(y_cord)])  # Adds the location to the quadrant one list
            if x_cord != 0:
                q_2.append([int(-x_cord), int(y_cord)])
            if y_cord != 0:
                q_3.append([int(-x_cord), int(-y_cord)])
            if x_cord != 0 and y_cord != 0:
                q_4.append([int(x_cord), int(-y_cord)])

    final = q_1 + q_2 + q_3 + q_4
    final.pop(0)
    # print("Total no.of nodes = ", len(final))

    # ---------------------------------------------
    final_list = list(final)
    final.append([0, 0])

    # ----------------Display NOde Details---------------_#
    # print("\n")
    # print('Nodes\' Details'.center(25, '-'))
    # print("%-5s %-4s %-20s"%("\nID", ":", "location\n"))

    for i in range(len(final_list)):
        nodes.append(node(final_list[i]))
        # print("%-4s %-4s %-20s"%(nodes[i].id, ":", nodes[i].location))

    return offset_dist, final_list, final


def engine(final_list, final, radio_range, offset_dist, randhops, no_of_packets, selectNodes, isForward=1,isTestBattery=False):
    attacker1 = attacker()

    no_of_SourceNodes = len(selectNodes)

    def index_to_locations(list1):
        source_node_pos = []
        for selectNode in selectNodes:
            req_location = nodes[selectNode].location
            source_node_pos.append(req_location)
            # print("\nSelected Source Node",selectNodes.index(selectNode)+1 ,"\nID : ", nodes[selectNode].id, "\nLocation : ", req_location)
        return source_node_pos

    source_node_positions = list(index_to_locations(selectNodes))

    # print(source_node_positions)

    def findIndex(cord):
        temp_index = final_list.index(cord)
        return temp_index

    def indexSearch(list1, isMin=1):
        if isMin:
            value = min(list1)
        else:
            value = max(list1)

        temp_index = list1.index(value)
        return temp_index

    def move_attacker(bigList, location, start=0):

        # print("Location = ", location, "\nstart", start)
        stored_index = 0
        req_cord = location
        # print("First req cord", req_cord)

        for listx in bigList:

            if location in listx:

                for ind in range(start, len(listx)):

                    if location == listx[ind] and not attacker1.isVisited(listx[ind - 1]):

                        if req_cord == location or ind < stored_index:
                            stored_index = ind
                            req_cord = listx[ind - 1]

                            # print("Storing cord ", req_cord, " at index ", ind)

        if req_cord != location:
            attacker1.getLocation(req_cord)
            # print("Recursion with cord", req_cord, "and stop", stored_index)
            return move_attacker(bigList, req_cord, stored_index)
        else:
            # print("Function ended at ", req_cord, " and start = ", start, " stored index = ", stored_index)
            return req_cord

    # ------------Neighboour------------------#

    def neighbour(loc):

        x = loc[0]  # Stores X ordinate as x variable
        y = loc[1]  # Stores Y ordinate as y variable
        neigh_list = []
        for i in range(len(final)):
            dist = ((final[i][0] - x) ** 2 + (final[i][1] - y) ** 2) ** 0.5  # Distance between nodes
            if int(dist) <= radio_range:  # Checks if the node is in range or not
                if loc == final[i]:  # To avoid the display of inputted node/same node again
                    continue  # skips the value
                else:  # Executes the neighbours
                    neigh_list.append(final[i])

        return neigh_list

    def nodes_with_dist(neighbour_list):
        temp_dist = {}
        for x in neighbour_list:
            dist = math.sqrt(pow(x[0], 2) + pow(x[1], 2))
            x = tuple(x)
            temp_dist[x] = dist
        sorted_dist = {x: y for x, y in sorted(temp_dist.items(), key=lambda item: item[1])}
        return sorted_dist

    # def randomPath(neighbour_list,isForward=1):
    #     sorted_dictionary = nodes_with_dist(neighbour_list)
    #     req_index = round(len(sorted_dictionary) / 2, 0)
    #     if isForward:
    #         num = random.randrange(0, req_index)
    #     else:
    #         num = random.randrange(req_index, len(sorted_dictionary))
    #     ans = list(sorted_dictionary)[num]
    #     ans = list(ans)
    #
    #     return ans
    #
    def findHopCount(cord):

        index = findIndex(cord)

        if nodes[index].knowHopCount:
            return nodes[index].hopCount

        x = abs(cord[0])
        y = abs(cord[1])

        if x > y:
            ord = x
        else:
            ord = y

        reqHopCount = ord / offset_dist

        nodes[index].getHopCount(reqHopCount)

        return reqHopCount

    def randomPath(neighbour_list, sendingNode, isForward=1):

        sHpCount = findHopCount(sendingNode)

        forwardingList = []
        for neighbour in neighbour_list:

            hpCount = findHopCount(neighbour)

            if isForward:
                if hpCount <= sHpCount:
                    forwardingList.append(neighbour)
            else:
                if hpCount >= sHpCount:
                    forwardingList.append(neighbour)

        ans = random.sample(forwardingList, 1)

        return ans[0]

    # ------ hop count control ---#

    def hop_control(ord):

        dist_list = []
        count = 0
        cord = ord
        nodes[findIndex(ord)].getType("Source Node")
        bigList = []

        while 1:

            index = findIndex(cord)

            if nodes[index].status == "Inactive":
                # print("Inactive Node Selected, Message  sent but not reached.")
                stop = True
                dist_list.pop()
                count -= 1
                break

            temp_neigh = neighbour(cord)  # temporary variable to store list of neighbour nodes

            if [0, 0] in temp_neigh:
                if count < randhops:

                    ans = randomPath(temp_neigh, cord, isForward)

                else:
                    ans = [0, 0]

            else:

                if count < randhops:

                    ans = randomPath(temp_neigh, cord, isForward)

                else:

                    ans = list(list(nodes_with_dist(temp_neigh))[0])

            reqdist = round(math.sqrt(pow((ans[0] - cord[0]), 2) + pow((ans[1] - cord[1]), 2)), 2)
            dist_list.append(reqdist)

            ## ENERGY CALCULATION

            Eelec, l, Eamp, Efs, d0 = node.giveEnergyDetails()

            if reqdist <= d0:
                sendEnergy = Eelec * l + Efs * l * pow(reqdist, 2)
            else:
                sendEnergy = Eelec * l + Eamp * l * pow(reqdist, 4)

            if count != 0:
                receiveEnergy = Eelec * l
            else:
                receiveEnergy = 0

            tranferEnergy = sendEnergy + receiveEnergy  # total energy utilised

            temp_remEnergy = nodes[index].energy - tranferEnergy  # updating the remaining energy

            nodes[index].getremEnergy(temp_remEnergy)

            ## END ENERGY CALCULAIION

            if nodes[index].energy <= 0:
                if nodes[index].energy < 0:
                    stop = True
                    nodes[index].getStatus()
                    dist_list.pop()
                    break
                else:
                    nodes[index].getStatus()
                    stop = True
                    break

            nodes[index].packetsent()
            if nodes[index].type != "Source Node":
                nodes[index].getType("Carrier Node")

            if cord not in bigList:
                nodes[index].getUniquePacket()

            count += 1

            # print("From : ", cord, "to : ", ans)

            bigList.append(cord)

            if ans == [0, 0]:
                stop = False
                break

            cord = ans

        return [sum(dist_list), count, stop, bigList]

    # ---------------------Packet count control-----------------------#

    def packet_control(no_of_packets):

        attacker.getSNLocations(source_node_positions)

        totalHops = 0

        for i in range(no_of_packets):

            if i == 0:

                temp_dist = []
                temp_hops = []
                temp_stop = []
                imp_list = []
                for source_node_position in source_node_positions:
                    # print("\nAttacker current position : ", attacker1.location)
                    li = hop_control(source_node_position)

                    temp_dist.append(li[0])
                    temp_hops.append(li[1])
                    totalHops += li[1]
                    temp_stop.append(li[2])
                    imp_list.append(li[3])

                if not isTestBattery:
                    attacker1.clearLocations()
                    # print(temp_hops)
                    # print(imp_list)
                    attacker1.getLocation(imp_list[indexSearch(temp_hops)][-1])
                    # print(attacker1.location)

                # print("\nDistance each Package travelled : ", temp_dist)
                # print("No.of Hops each Package travelled : ", temp_hops)

            else:

                temp_dist = []
                temp_hops = []
                temp_stop = []
                imp_list = []
                for source_node_position in source_node_positions:
                    # print("\nAttacker current position : ", attacker1.location)
                    li = hop_control(source_node_position)

                    temp_dist.append(li[0])
                    temp_hops.append(li[1])
                    totalHops += li[1]
                    temp_stop.append(li[2])
                    imp_list.append(li[3])

                if not isTestBattery:
                    # print("\nCalling move attacker function")

                    move_attacker(imp_list, attacker1.location)

                    # print("FUnction ended")

            if attacker1.isFoundSN() or True in temp_stop:
                # print("\n")
                if True in temp_stop:
                    packets_sent = i * no_of_SourceNodes
                    result = "Energy"
                    # print("> Node Energy over <".center(50, "-"))
                    # print("\t")
                    # print("> Package Transfer Stopped <".center(50, "="))
                    # print("\nNo.of packages reached Base Station : ", packets_sent)
                else:
                    result = "Attacker"
                    packets_sent = (i + 1) * no_of_SourceNodes
                    if isTestBattery:
                        print("result")
                    # print("> Attacker Found Source Node <".center(50, "-"))
                    # print("\t")
                    # print("> Package Transfer Stopped <".center(50, "="))
                    # print("\nNo.of packages reached Base Station : ", packets_sent)

                # print("\nDistance last Package travelled : ", temp_dist)
                # print("No.Of Hops last package travelled : ", temp_hops)

                safety_period = (packets_sent / (no_of_packets * no_of_SourceNodes)) * 100
                break

        else:
            packets_sent = no_of_packets * no_of_SourceNodes
            safety_period = 100
            result = "Successful"
            # print("\t")
            # print("> All {} Packets reached Bases Station <".format(packets_sent).center(50, "-"))

        # print("\ntotal HOps = ", totalHops)

        return temp_hops, temp_dist, safety_period, packets_sent, result, totalHops

    hop_count, dist_count, safety_period, sentPackets, result, totalHops = packet_control(no_of_packets)
    # print("\nSafety Period = {}".format(safety_period))

    avgHops = round(totalHops / sentPackets, 0)

    # ----------Calculating Entropy--------------#

    def findEntropy():

        Entropy = 0
        for i in range(len(nodes)):
            if nodes[i].type == "Carrier Node":
                probab = nodes[i].sentpackets / sentPackets  # packets carrier node sent / packets source node sent
                nodes[i].getProbability(probab)
                halfEnt = probab * math.log2(1/probab)
                Entropy += halfEnt
        return Entropy

    entropy = findEntropy()

    # print("\nThe Entropy of Network ; " + str(entropy))

    # --------Calculating Energy--------------#

    def EnergyDetails():

        def EnergyRem():
            EnergyRemaining = 0
            for i in range(len(nodes)):
                EnergyRemaining += nodes[i].energy
            return EnergyRemaining

        EnergyRemaining = EnergyRem()

        def EnergyTot():
            EnergyTotal = nodes[0].initialEnergy * len(nodes)
            return EnergyTotal

        EnergyTotal = EnergyTot()

        EnergyConsumed = EnergyTotal - EnergyRemaining

        # print("Total Energy = {} nJ\nEnergy Remaining = {} nJ\nEnergy Consumed = {} nJ".format(EnergyTotal,EnergyRemaining,EnergyConsumed))

        return EnergyConsumed

    energyConsumed = EnergyDetails()

    # ---------------------display node energy details-------#
    def display_details():

        print('* Nodes\' Details *'.center(127, '-'))
        print("%-5s %-5s %-19s %-4s %-15s %-4s %-20s %-4s %-10s %-4s %-15s %-4s %-4s %-4s %-4s" % (
            "\nID", ":", "Location", ":", "Initial Energy", ":", "Final Energy", ":", "Status", ":", "Node Type", ":",
            "Packets Sent", ":", "Unique Packets Sent\n"))
        for i in range(len(nodes)):
            print("%-4s %-4s %-20s %-4s %-15s %-4s %-20s %-4s %-10s %-4s %-15s %-4s %-15s %-4s %-4s" % (
                nodes[i].id, ":", nodes[i].location, ":", nodes[i].initialEnergy, ":", nodes[i].energy, ":",
                nodes[i].status,
                ":", nodes[i].type, ":", nodes[i].sentpackets, ":", nodes[i].uniquePackets))

    # display_details()

    nodes.clear()  # CLear the Nodes list

    return hop_count, dist_count, result, safety_period, energyConsumed, avgHops, entropy,sentPackets


def simulation_control(no_of_simulations, length, selectNodes, csv_path, isCreate, isUpload, scenario, areaName,isTestBattery):
    # ---Required Details--- #
    # length = 500
    breadth = length
    radio_range = 71
    randhops = 10
    no_of_packets = 10000
    area = round(length * breadth, 2)
    # selectNodes = [27,31]
    nature = "backward"

    def select(nature):
        if nature == "coin":
            isForward = random.randint(0, 1)

        elif nature == "forward":
            isForward = 1
        else:
            isForward = 0
        return isForward

    safety_periods = []
    consumed_energies = []

    energy_failed = 0
    attacker_succeed_count = 0

    offsetDist, final_list, final = square(length, breadth, radio_range)

    type = ["TYPE", "Simulations", "Length", "Breadth", "Area", "Radio range", "Offset Distance", "packets sending",
            "Randomhops"]
    data = ["DATA", no_of_simulations, length, breadth, area, radio_range, offsetDist, no_of_packets, randhops]

    data1t = []
    data1d = []
    for selectNode in selectNodes:
        top = ["ID", "Location"]
        bottom = [nodes[selectNode].id, nodes[selectNode].location]
        data1t.append(top)
        data1d.append(bottom)

    type.extend(data1t)
    data.extend(data1d)

    sumAvgHops = 0
    sumEntropy = 0
    sumSentPackets = 0

    barName = str(scenario) + "-" + str(areaName) + " : "
    for simulation in tqdm(range(no_of_simulations), desc=barName, ascii=False, ncols=100, leave=False):

        isForward = select(nature)

        if simulation != 0:
            nodes.clear()
            for i in range(len(final_list)):
                nodes.append(node(final_list[i]))

        # print("\n")
        # print("\\ SIMULATION - {} /".format(simulation + 1).center(150, "*"))
        # print(nature, "=", isForward)

        hopCount, distCount, result, safety_period, energy_consumed, avgHops, entropy,sentPackets = engine(final_list, final,
                                                                                               radio_range, offsetDist,
                                                                                               randhops, no_of_packets,
                                                                                               selectNodes, isForward,isTestBattery)

        sumAvgHops += avgHops
        sumEntropy += entropy
        sumSentPackets += sentPackets

        path_name = lambda x: "Forward" if x else "Backward"

        data2t = ["Path Type", "Hops", "Distance", "Safety Period", "Energy Consumed", "Average Hops", "Entropy","Packets Sent","Result"]
        data2d = [path_name(isForward), hopCount, distCount, safety_period, energy_consumed, avgHops, entropy,sentPackets,result]
        type.extend(data2t)
        data.extend(data2d)

        safety_periods.append(safety_period)
        consumed_energies.append(energy_consumed)

        if result == "Attacker":
            attacker_succeed_count += 1
        elif result == "Energy":
            energy_failed += 1

    attacker_failed_count = no_of_simulations - attacker_succeed_count
    capture_ratio = attacker_succeed_count / no_of_simulations
    capture_percentage = capture_ratio * 100

    safety_period_avg = sum(safety_periods) / len(safety_periods)
    energy_consumed_avg = sum(consumed_energies) / len(consumed_energies)
    energy_consumed_avg_J = round(energy_consumed_avg * 0.0000000010, 4)
    # print(sumAvgHops)
    # print(sumEntropy)
    # print("----afa")
    avgAvgHops = sumAvgHops / no_of_simulations
    avgEntropy = sumEntropy / no_of_simulations
    avgSentPackets = sumSentPackets / no_of_simulations

    data3t = ["Average Packets Sent","Average Entropy", "Average of Average Hops", " Average Safety Period", "Average Energy Consumed (nJ)",
              "Average Energy Consumed (J)", "Times Attacker Succeeded", "Capture Ratio", "Capture Percentage"]
    data3d = [avgSentPackets,avgEntropy, avgAvgHops, safety_period_avg, energy_consumed_avg, energy_consumed_avg_J,
              attacker_succeed_count, capture_ratio, capture_percentage]
    type.extend(data3t)
    data.extend(data3d)

    def csv_req():
        time_at_instant = datetime.now()
        time_at_instant_str = str(time_at_instant)

        x = time.time()  # Gives time in seconds
        dt = datetime.today()  # Gives Date
        y = dt.day  # Returns only day from Date

        unique_code_str = str(x / y * 10000)

        return unique_code_str, time_at_instant_str

    def csv_create(column1, column2, path, isNew=0):

        unique_code_str, time_at_instant_str = csv_req()

        if isNew:
            df01 = {
                unique_code_str: column1,
                time_at_instant_str: column2
            }
            dfinput = pd.DataFrame(df01)
        else:
            dfinput = pd.read_csv(path)
            dfinput[unique_code_str] = column1
            dfinput[time_at_instant_str] = column2

        dfinput.to_csv(path, index=False)

    # csv_path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_multiple_sources_e.csv'
    if isUpload:
        csv_create(type, data, csv_path, isCreate)

    # print("\n")
    # print("\\ SIMULATION ENDED /".center(150, "*"))
    # print("\t")
    # print("* Results *".center(150, "-"))
    # print("\nNo.of times attacker succeed : ", attacker_succeed_count)
    # print("No.of times attacker failed : ", attacker_failed_count)
    # print("\nAverage Entropy :  ",avgEntropy)
    # print("\nAverage Hop count : ",avgAvgHops)
    # print("\nCapture Ratio : ", capture_ratio)
    # print("Capture Percentage : ", capture_percentage,"%")
    # print("\nSafety Periods list = {}\nSafety Period Count = {}\nSafety Period average = {}".format(safety_periods,len(safety_periods),safety_period_avg))
    # print("\nEnergy Consumed list = {}\nEnergy Consumed average = {}".format(consumed_energies,energy_consumed_avg))




def input_control():
    isUpload = True
    isSingle = False
    isTestBattery = True
    no_of_simulations = 1
    baseLenghts = 500
    baseSNlOcs = [[52, 62],
                  [64, 119],
                  [64, 94, 119],
                  [64, 94, 119, 34]]
    singlebaseSNlOcs = [[64]]
    area = 25

    lengths = [707.10678118655, 866.02540378444, 1000, 1118.0339887499, 1224.7448713916, 1414.2135623731]

    sourceNodeLocs = [[[116, 102], [149, 133], [227, 207], [272, 250], [321, 297], [431, 403]],  # b
                      [[118, 223], [151, 287], [229, 439], [274, 527], [323, 623], [433, 839]],  # c
                      [[118, 174, 223], [151, 223, 287], [229, 339, 439], [274, 527, 406], [323, 623, 479],
                       [433, 839, 643]],
                      [[118, 174, 223, 62], [151, 223, 287, 79], [229, 339, 439, 119], [274, 527, 406, 142],
                       [323, 623, 479, 167], [433, 839, 643, 223]]]
    singleSourceNodeLocs = [[118], [151], [229], [274], [323], [433]]
    areas = [50, 75, 100, 125, 150, 200]

    if isSingle:
        count = 1
    else:
        count = len(sourceNodeLocs)

    print("\nBackward Walk With Shortest Path Simulation\nPROCESS STARTED\n")
    for a in tqdm(range(count), desc="Process : ", ascii=False, ncols=100, miniters=1):

        if isSingle:
            path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_single_source.csv'
            scenario = "S"

            simulation_control(no_of_simulations, baseLenghts, singlebaseSNlOcs[a], path, 1, isUpload, scenario, area,isTestBattery)

            for i, length in tqdm(enumerate(lengths), desc="Process : ", ascii=False, ncols=100, miniters=1):
                simulation_control(no_of_simulations, length, singleSourceNodeLocs[i], path, 0, isUpload, scenario,
                                   areas[i],isTestBattery)

        else:
            if a == 0:
                path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_multiple_sources_b.csv'
            elif a == 1:
                path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_multiple_sources_c.csv'
            elif a == 2:
                path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_multiple_sources_d.csv'
            else:
                path = '/Users/apple/Project_1/Observations/Data_Collected/New_Data/Backward_Walk/fbackward_multiple_sources_e.csv'

            scenario = path[-5].upper()

            simulation_control(no_of_simulations, baseLenghts, baseSNlOcs[a], path, 1, isUpload, scenario, area,isTestBattery)

            for i, length in enumerate(lengths):
                simulation_control(no_of_simulations, length, sourceNodeLocs[a][i], path, 0, isUpload, scenario, areas[i],isTestBattery)




    print("PROCESS ENDED SUCCESSFULLY")


input_control()

final = time.perf_counter()
print(f"finished in {round(final - start, 2)} seconds")



