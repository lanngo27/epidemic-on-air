import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as md
import scipy.stats
import random
import datetime as dt
from math import isinf
import si_animator as animator
import operator

def SI(network, sorted_flights, start_node, p, immunized_nodes=set()):
    infected_time = [float('inf')] * network.number_of_nodes()

    for flight in sorted_flights:
        source = flight["Source"]
        start_time = flight["StartTime"]
        if source == start_node:
            infected_time[source] = start_time
            break

    for flight in sorted_flights:
        source = flight["Source"]
        destination = flight["Destination"]
        start_time = flight["StartTime"]
        end_time = flight["EndTime"]

        if destination not in immunized_nodes and infected_time[source] <= start_time and infected_time[
            destination] > end_time:
            rand_num = random.random()
            if rand_num < p:
                infected_time[destination] = end_time

    return infected_time

def SI_link(network, sorted_flights, start_node, p):
    infected_time = [float('inf')] * network.number_of_nodes()

    for flight in sorted_flights:
        source = flight["Source"]
        start_time = flight["StartTime"]
        if source == start_node:
            infected_time[source] = start_time
            break

    edgelist = network.edges()
    link_list=[0]*network.number_of_edges()
    infected_destination=[None]*network.number_of_nodes()

    for flight in sorted_flights:
        source = flight["Source"]
        destination = flight["Destination"]
        start_time = flight["StartTime"]
        end_time = flight["EndTime"]

        if infected_time[source] <= start_time and infected_time[destination] > end_time:
            rand_num = random.random()
            if rand_num < p:
                infected_time[destination] = end_time
                infected_destination[destination] = source

    for idx, des in enumerate(infected_destination):
        if des is not None:
            link = (idx, des)
            infected_idx = -1
            if network.has_edge(idx, des):
                if link in edgelist:
                    infected_idx = edgelist.index(link)
                    link_list[infected_idx] = 1
                else:
                    infected_idx = edgelist.index((des, idx))
                    link_list[infected_idx] = 1
    return link_list

def get_link_weights(network):
    weights = []
    edge_list = network.edges(data = True)
    for(i, j, data) in edge_list:
        weights.append(data['weight'])

    return weights

def get_link_overlap(network):
    """
    Calculates link overlap:
    O_ij = n_ij / [(k_i - 1) + (k_j - 1) + 1]

    """

    overlaps = []
    for edge in network.edges():
        i = edge[0]
        j = edge[1]
        k_i = network.degree(i)
        k_j = network.degree(j)
        common_neighbor = set.intersection(set(network.neighbors(i)), set(network.neighbors(j)))
        n_ij = len(common_neighbor)

        if k_i + k_j == n_ij + 2:
            O_ij = 0
        else:
            O_ij = n_ij/(k_i + k_j - 2 - n_ij)
        overlaps.append(O_ij)
    return overlaps


def prevalence(infected_times, num_iter, num_of_nodes, time_step):
    prevalence = [0] * num_of_nodes

    for t in range(num_of_nodes):
        time = time_step[t]
        num_infected = 0

        for i in range(num_iter):
            for item in infected_times[i]:
                if item<time:
                    num_infected+=1

        num_infected=num_infected/num_iter
        prevalence[t]=num_infected/num_of_nodes
    return prevalence

def immunization(network, sorted_flights):
    p = 0.5
    colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-', 'k-', '#50f386']
    num_iter = 20
    num_nodes = network.number_of_nodes()

    kshell = sorted(nx.find_cores(network).items(), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_kshell = {tup[0] for tup in kshell}
    clustering = sorted(nx.clustering(network).items(), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_clustering = {tup[0] for tup in clustering}
    degree = sorted(list(nx.degree(network).items()), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_degree = {tup[0] for tup in degree}
    strength = sorted(list(nx.degree(network, weight='weight').items()), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_strength = {tup[0] for tup in strength}
    betweenness = sorted(nx.betweenness_centrality(network, normalized=True).items(), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_betweenness = {tup[0] for tup in betweenness}
    closeness = sorted(nx.closeness_centrality(network, distance='weight').items(), key=operator.itemgetter(1), reverse=True)[0:10]
    immunized_node_closeness = {tup[0] for tup in closeness}

    immunized_node_random = set(np.random.randint(0, num_nodes, 10))

    # social network strategy
    immunized_node_social = set()
    while len(immunized_node_social) < 10:
        random = np.random.randint(0, num_nodes, 10)
        for node in random:
            rnd_neighbor = np.random.choice(list(network.neighbors(node)))
            immunized_node_social.add(rnd_neighbor)


    all_immunized_nodes = immunized_node_social\
                            .union(immunized_node_kshell)\
                            .union(immunized_node_clustering)\
                            .union(immunized_node_degree)\
                            .union(immunized_node_strength)\
                            .union(immunized_node_betweenness)\
                            .union(immunized_node_closeness)\
                            .union(immunized_node_random)
    # *************
    available_nodes = list(set(range(0, num_nodes)) - all_immunized_nodes)
    seed_list = np.random.choice(available_nodes, 20)

    first_departure = sorted_flights[0]["StartTime"]
    last_arrival = max(sorted_flights["EndTime"])
    num_nodes = network.number_of_nodes()

    fig5 = plt.figure()
    ax = fig5.add_subplot(111)
    plt.xticks(rotation=45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Prevalence")

    time_step = np.linspace(first_departure, last_arrival, num_nodes)
    time_step_converted = [dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M') for date in time_step]
    time_step_converted = md.datestr2num(time_step_converted)

    strategies = {
        'kshell': immunized_node_kshell,
        'clustering': immunized_node_clustering,
        'degree': immunized_node_degree,
        'strength': immunized_node_strength,
        'betweenness': immunized_node_betweenness,
        'closeness': immunized_node_closeness,
        'random': immunized_node_random,
        'social': immunized_node_social
    }

    for idx, strategy in enumerate(strategies.keys()):
        immunized_nodes = strategies[strategy]
        infected_times = []

        for i in range(num_iter):
            seed_node = seed_list[i]
            infected = SI(network, sorted_flights, seed_node, p, immunized_nodes)
            infected.sort()
            infected_times.append(infected)

        avg_prevalence = prevalence(infected_times, num_iter, num_nodes, time_step)

        ax.plot_date(time_step_converted, avg_prevalence, colors[idx], label=strategy)

    ax.legend(loc=0)
    fig5.tight_layout()
    fig5.savefig('./immunization.png')
    return None

if __name__ == "__main__":
    traffic_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names=True, dtype=int)

    #Flights sorted according to start time
    sorted_flights = np.copy(traffic_data)
    sorted_flights.sort(order=['StartTime'])

    network = nx.read_weighted_edgelist('aggregated_US_air_traffic_network_undir.edg', nodetype=int)
    num_of_nodes = network.number_of_nodes()

    #fig1 = plt.figure()
    #ax = fig1.add_subplot(111)
    #nx.draw_spring(network, node_size=10)
    #fig1.savefig('./traffic_visualization.png')

    # Task 1
    p = 1
    start_node = 0
    infected = SI(network, sorted_flights, start_node, p)
    print("Allentown is infected at: %s, Anchorage (node 41) is infected at: %s" % (infected[0], infected[41]))

    # Task 2
    infection_prob = [0.01, 0.05, 0.1, 0.5, 1.0]
    colors = ['r-', 'g-', 'b-', 'c-', 'm-']
    first_departure = sorted_flights[0]["StartTime"]
    last_arrival = max(sorted_flights["EndTime"])

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Prevalence p(t)")
    time_step = np.linspace(first_departure, last_arrival, num_of_nodes)
    time_axis = [dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M') for date in time_step]
    time_axis = md.datestr2num(time_axis)

    for idx, p in enumerate(infection_prob):
        infected_list = []
        for n in range(10):
            infected = SI(network, sorted_flights, start_node, p)
            infected.sort()
            infected_list.append(infected)

        avg_prevalence = [0] * num_of_nodes

        for t in range(num_of_nodes):
            time = time_step[t]
            num_infected = 0

            for i in range(10):
                for item in infected_list[i]:
                    if item < time:
                        num_infected += 1

            num_infected = num_infected / 10
            avg_prevalence[t] = num_infected / num_of_nodes

        ax.plot_date(time_axis, avg_prevalence, colors[idx], label="p = " + str(p))

    ax.legend(loc=0)
    fig2.tight_layout()
    fig2.savefig('./average_prevalence.png')

    # Task 3
    start_node = [0, 4, 41, 100, 200]
    p = 0.1

    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Prevalence p(t)")
    time_step = np.linspace(first_departure, last_arrival, num_of_nodes)
    time_axis = [dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M') for date in time_step]
    time_axis = md.datestr2num(time_axis)

    for idx, node in enumerate(start_node):
        infected_list = []
        for n in range(10):
            infected = SI(network, sorted_flights, node, p)
            infected.sort()
            infected_list.append(infected)

        avg_prevalence = [0] * num_of_nodes

        for t in range(num_of_nodes):
            time = time_step[t]
            num_infected = 0

            for i in range(10):
                for item in infected_list[i]:
                    if item < time:
                        num_infected += 1

            num_infected = num_infected / 10
            avg_prevalence[t] = num_infected / num_of_nodes

        ax.plot_date(time_axis, avg_prevalence, colors[idx], label="node-ids = " + str(node))

    ax.legend(loc=0)
    fig3.tight_layout()
    fig3.savefig('./average_prevalence_2.png')

    # Task 4
    start_node = np.random.randint(0, num_of_nodes, 20)
    infected_list = []
    for idx, node in enumerate(start_node):
        infected_link = list()
        infected = SI(network, sorted_flights, node, 0.5)
        infected_list.append(infected)

    kshell = np.array(list(nx.find_cores(network).values()))
    clustering = np.array(list(nx.clustering(network).values()))
    degree = np.array(list(nx.degree(network).values()))
    strength = np.array(list(nx.degree(network, weight='weight').values()))
    betweenness = np.array(list(nx.betweenness_centrality(network, normalized=True).values()))
    closeness = np.array(list(nx.closeness_centrality(network).values()))
    centrality = np.array([kshell, clustering, degree, strength, betweenness, closeness])

    start_node = np.random.randint(0, num_of_nodes, 50)

    infected_list = []
    for idx, node in enumerate(start_node):
        infected = SI(network, sorted_flights, node, 0.5)
        infected_list.append(infected)

    infected_list_np = np.array(infected_list)
    median_infection = []
    for i in range(num_of_nodes):
        node_simulation = infected_list_np[:, i]
        infected_count = 0
        for time in node_simulation:
            if not isinf(time):
                infected_count += 1
        print(infected_count)
        if infected_count >= 25:
            median_infection.append(np.median(node_simulation))
        else:
            np.delete(centrality, i, axis=1)

    y_label = 'Median infection times'
    scatter_base_path = 'scatter'
    titles = ['k-shell', 'Clustering', 'Degree', 'Strength', 'Betweenness', 'Closeness']

    for i in range(6):
        x_values = centrality[i, :]
        x_label = titles[i]
        scatter_path = scatter_base_path + '_' + titles[i] + '.png'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_values, median_infection)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        fig.tight_layout()
        fig.savefig(scatter_path)
        print('Scatter plot ready!')

        spearmanr = scipy.stats.spearmanr(median_infection, x_values)
        print('Spearman coefficient between infection times and ' + titles[i] + ': ' + str(spearmanr[0]))

    # Task 5
    immunization(network, sorted_flights)

    #Task 6
    weights = get_link_weights(network)
    overlaps = get_link_overlap(network)
    e_betweenness = list(nx.edge_betweenness_centrality(network, normalized=True).values())
    link_centrality = np.array([weights, overlaps, e_betweenness])

    infected_link = []
    start_node = np.random.randint(0, num_of_nodes,20)
    for i,node in enumerate(start_node):
        infected = SI_link(network, sorted_flights, node, 0.5)
        infected_link.append(infected)

    infected_link_np = np.array(infected_link)
    f_ij = []
    num_of_edges=network.number_of_edges()
    for i in range(num_of_edges):
        f_ij.append(np.mean(infected_link_np[:, i]))

    #maximum spanning tree
    net = network.copy()
    for u, v, d in net.edges(data=True):
        d['weight'] = -1 * d['weight']
    maximal = nx.minimum_spanning_tree(net, weight='weight')
    for u, v, d in maximal.edges(data=True):
        d['weight'] = -1*d['weight']


    #visualization
    id_data = np.genfromtxt('US_airport_id_info.csv', delimiter=',', dtype=None, names=True)
    xycoords = {}
    for row in id_data:
        xycoords[row['id']] = (row['xcoordviz'], row['ycoordviz'])

    fig6a, ax6a = animator.plot_network_usa(maximal, xycoords, edges=maximal.edges())
    fig6a.savefig('air_maxtree.png')
    fig6b, ax6b = animator.plot_network_usa(network, xycoords, edges=network.edges(),linewidths=f_ij)
    fig6b.savefig("air_maps.png")

    #scatter plot
    y_label = 'Fraction of times a link is used for infecting the disease'
    scatter_base_path = 'scatter'
    titles = ['Link weight', 'Link neighborhood overlap', 'Link betweenness centrality']

    for i in range(3):
        x_values = link_centrality[i, :]
        x_label = titles[i]
        scatter_path = scatter_base_path + '_' + titles[i] + '.png'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_values, f_ij)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()

        fig.tight_layout()
        fig.savefig(scatter_path)
        print('Scatter plot ready!')

        spearmanr = scipy.stats.spearmanr(f_ij, x_values)
        print('Spearman coefficient between f_ij and ' + titles[i] + ': ' + str(spearmanr[0]))
