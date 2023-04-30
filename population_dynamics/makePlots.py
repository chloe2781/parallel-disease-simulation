import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_color_and_size_scale():
    with open('points.txt', 'r') as file:
        # Initialize an empty list to store the tuples
        tuple_list = []
        # Iterate through each line in the file
        for line in file:
            # Convert the stringified tuple into a tuple object
            tuple_str = line.strip()  # Remove any leading/trailing whitespaces
            tuple_obj = tuple(map(int, tuple_str[1:-1].split(',')))  # Convert to tuple
            # Append the tuple to the list
                #edge cases are removed due to collection of people at graph boundaries
            rescaled_x = int(tuple_obj[0]/3)
            rescaled_y = int(tuple_obj[1]/3)
            tuple_list.append((rescaled_x, rescaled_y))

    #Graphing with color scale by number of COVID deaths
    freq = Counter(tuple_list)
    # create a list of alpha values based on frequency
    alpha_values = [1 / freq[coord] for coord in tuple_list]
    # create a colormap ranging from light to dark shades of red
    cmap = plt.get_cmap('Reds')
    # normalize the frequencies to the range [0,1]
    norm = plt.Normalize(0, max(freq.values()))
    size_values = [freq[coord] * 25 for coord in tuple_list]
    # create a list of colors based on the normalized frequencies
    colors = [cmap(norm(freq[coord])) for coord in tuple_list]
    # plot the coordinates with varying alpha values and colors
    plt.scatter([coord[0] for coord in tuple_list], [coord[1] for coord in tuple_list], c=colors, s=size_values, alpha=alpha_values)

    # set the title and axis labels
    plt.title('COVID-19 Deaths By Location')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # show the plot
    plt.show()


#plot_color_and_size_scale()


