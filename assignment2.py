# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import networkx as nx

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

def plot_block(coordinates):
    fig, ax = plt.subplots()

    for coordinate in coordinates:
        block_id, width, height, x1, y1 = coordinate
        ax.add_patch(plt.Rectangle((x1, y1), width, height, edgecolor='black', facecolor='none'))
        # ax.text(x + width / 2, y + height / 2, f"Block [len(ax.patches)}", ha='center', va='center')
        plt.text(x1+width/2, y1+height/2, block_id)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('VLSI Layout')
    ax.set_xlim(-45, 45)
    ax.set_ylim(-60, 60)
    plt.show()

class Node:
  def __init__(self, data):
    self.data = data
    self.left = None  # Reference to the left child node
    self.right = None  # Reference to the right child node
    self.parent = None
    self.xl = 0
    self.yl = 0
    self.xr = 0
    self.yr = 0

  def __str__(self):
    return str(self.data)

def read_text_file(filename):
    blocks = []
    with open(filename, 'r') as file:
        for line in file:
            row = [int(x.strip()) for x in line.split(',')]
            blocks.append(row)
    return blocks

def read_polish_expression(filename):
    count_operator = 0
    count_operand = 0
    with open(filename, 'r') as file:
        input_string = file.read().strip()
    polish_expression = []
    for char in input_string:
        if char.isdigit():
            polish_expression.append(int(char))
        else:
            polish_expression.append(char)
    for i in range(0, len(polish_expression)):
        if(type(polish_expression[i]) is int):
            count_operand = count_operand + 1
        else:
            count_operator = count_operator + 1
    if(count_operand == count_operator):
        raise ValueError("Invalid Polish expression")
    return polish_expression

def read_wire_length_matrix(filename):
    wire_length_matrix = []
    with open(filename, 'r') as file:
        for line in file:
            row = [int(x) for x in line.split()]
            wire_length_matrix.append(row)
    return wire_length_matrix

size_of_blocks = read_text_file('blocks.txt')
polish_expression = read_polish_expression('polish_expression.txt')
wire_length_matrix = read_wire_length_matrix('wire_length_matrix.txt')

def check_input_files(size_of_blocks, wire_length_matrix):
    if(len(size_of_blocks) != len(wire_length_matrix)):
        raise ValueError("Invalid input files. Number of blocks and wire length matrix do not match.")
check_input_files(size_of_blocks, wire_length_matrix)

coordinates = list()
stack1 = Stack()

def ptog(polish_expression):
  for ch in polish_expression:
    if(type(ch) is int):
      node = Node(ch)
      stack1.push(node)
    elif(ch == 'H' or ch == 'V'):
      node = Node(ch)
      node.right = stack1.pop()
      node.right.parent = node
      node.left = stack1.pop()
      node.left.parent = node
      stack1.push(node)
  return node
print(ptog(polish_expression))
node = ptog(polish_expression)

x1, y1 = 0, 0
coordinates1 = list()
def plot(node, coordinates):
  if(node.left == None and node.parent.left.data == node.data and node.parent.data == 'V'):
    x1 = node.parent.xl
    y1 = node.parent.yl
    node.xl = x1
    node.yl = y1
    node.xr = x1 + size_of_blocks[node.data-1][1]
    node.yr = y1 + size_of_blocks[node.data-1][2]
    coordinates.append((node.data, size_of_blocks[node.data-1][1], size_of_blocks[node.data-1][2], node.xl, node.yl))
    node.parent.xl = node.xl + size_of_blocks[node.data-1][1]
    node.parent.yl = node.yl
    node.parent.xr = node.xr
    node.parent.yr = node.yr
    return
  elif(node.left == None and node.parent.left.data == node.data and node.parent.data == 'H'):
    x1 = node.parent.xl
    y1 = node.parent.yr
    node.xl = x1
    node.yl = y1
    node.xr = x1 + size_of_blocks[node.data-1][1]
    node.yr = y1 + size_of_blocks[node.data-1][2]
    coordinates.append((node.data, size_of_blocks[node.data-1][1], size_of_blocks[node.data-1][2], node.xl, node.yl))
    node.parent.xl = node.xl
    node.parent.yl = node.yl + size_of_blocks[node.data-1][2]
    node.parent.xr = node.xr
    node.parent.yr = node.yr
    return
  elif(node.right == None and node.parent.right.data == node.data and node.parent.data == 'V'):
    x1 = node.parent.xl
    y1 = node.parent.yl
    node.xl = x1
    node.yl = y1
    node.xr = x1 + size_of_blocks[node.data-1][1]
    node.yr = y1 + size_of_blocks[node.data-1][2]
    coordinates.append((node.data, size_of_blocks[node.data-1][1], size_of_blocks[node.data-1][2], node.xl, node.yl))
    node.parent.xl = node.xl
    node.parent.yl = node.yl
    node.parent.xr = node.xr
    node.parent.yr = max(node.parent.left.yr, node.yr)
    return
  elif(node.right == None and node.parent.right.data == node.data and node.parent.data == 'H'):
    x1 = node.parent.xl
    y1 = node.parent.yr
    node.xl = x1
    node.yl = y1
    node.xr = x1 + size_of_blocks[node.data-1][1]
    node.yr = y1 + size_of_blocks[node.data-1][2]
    coordinates.append((node.data, size_of_blocks[node.data-1][1], size_of_blocks[node.data-1][2], node.xl, node.yl))
    node.parent.xl = node.xl
    node.parent.yl = node.yl
    node.parent.xr = max(node.parent.left.xr, node.xr)
    node.parent.yr = node.yr
    return
  if(node.left != None):
    if(node.parent != None):
      node.xl = node.parent.xl
      node.xr = node.parent.xr
      node.yl = node.parent.yl
      node.yr = node.parent.yr
    plot(node.left, coordinates)
  if(node.right != None and node.data == 'H'):
    node.xl = node.left.xl
    node.xr = node.left.xr
    node.yl = node.left.yr
    node.yr = node.left.yr
    plot(node.right, coordinates)
  elif(node.right != None and node.data == 'V'):
    node.xl = node.left.xr
    node.xr = node.left.xr
    node.yl = node.left.yl
    node.yr = node.left.yr
    plot(node.right, coordinates)
  if(node.right != None and node.left != None and (node.data == 'V' or node.data == 'H')):
    node.xl = node.right.xl
    node.xr = node.right.xr
    node.yl = node.right.yl
    node.yr = node.right.yr
  return coordinates
coordinates1 = plot(node, coordinates1)
print(coordinates1)
plot_block(coordinates1)

def routing_length(wire_length_matrix, coordinates):
  length = 0
  for i in range(0, len(wire_length_matrix)):
    for j in range(0, len(coordinates)):
      length = length + abs(((coordinates[i][1]/2 + coordinates[i][3]) - (coordinates[j][1]/2 + coordinates[j][3]))+((coordinates[i][2]/2 + coordinates[i][4]) - (coordinates[j][2]/2 + coordinates[j][4])))*wire_length_matrix[i][j]
  return length
# routing_length(wire_length_matrix, coordinates1)

def floorplan_area(coordinates):
  xmin = 1000000
  ymin = 1000000
  xmax = 0
  ymax = 0
  for i in range(0, len(coordinates)):
    if(coordinates[i][3] < xmin):
      xmin = coordinates[i][3]
    if(coordinates[i][4] < ymin):
      ymin = coordinates[i][4]
    if(coordinates[i][3] + coordinates[i][1] > xmax):
      xmax = coordinates[i][3] + coordinates[i][1]
    if(coordinates[i][4] + coordinates[i][2] > ymax):
      ymax = coordinates[i][4] + coordinates[i][2]
  return ((xmax-xmin)*(ymax-ymin))
# floorplan_area(coordinates1)

def cost_floorplan(floorplan_area, routing_length):
  return ((0.75*floorplan_area)+(0.25*routing_length))
# cost_floorplan(floorplan_area(coordinates1), routing_length(wire_length_matrix, coordinates1))

def m1(polish_expression):
  count = 0
  for i in range(0, len(polish_expression)-1):
    if(type(polish_expression[i]) is int and type(polish_expression[i+1]) is int):
      if(random.random() < 0.5 and count == 0):
        polish_expression[i] = polish_expression[i] + polish_expression[i+1]
        polish_expression[i+1] = polish_expression[i] - polish_expression[i+1]
        polish_expression[i] = polish_expression[i] - polish_expression[i+1]
        count = 1
  return polish_expression
# m1(polish_expression)

def m2(polish_expression):
  count = 0
  temp = ''
  for i in range(0, len(polish_expression)-1):
    if((polish_expression[i] == 'H' or polish_expression[i] == 'V') and (polish_expression[i+1] == 'H' or polish_expression[i+1] == 'V')):
      if(random.random() < 0.5 and count == 0):
        temp = polish_expression[i+1]
        polish_expression[i+1] = polish_expression[i]
        polish_expression[i] = temp
        count = 1
  return polish_expression
# m2(polish_expression)

def find_consecutive_repeated_letters(arr):
    for i in range(len(arr) - 1):
        if arr[i] == arr[i + 1] and arr[i] in ['H', 'V']:
            return 1
    return None

def m3(polish_expression):
  count = 0
  count_operator = 0
  count_operand = 0
  for i in range(0, len(polish_expression)-1):
    if(polish_expression[i] == 'V' or polish_expression[i] == 'H'):
      count_operator = count_operator + 1
    elif(type(polish_expression[i]) is int):
      count_operand = count_operand + 1
    if(((polish_expression[i] == 'H' or polish_expression[i] == 'V') and (type(polish_expression[i+1]) is int)) or (type(polish_expression[i]) is int and (polish_expression[i+1] == 'H' or polish_expression[i+1] == 'V'))):
      if(random.random() < 0.5 and count == 0 and count_operand-1 > count_operator+1):
        temp = polish_expression[i+1]
        polish_expression[i+1] = polish_expression[i]
        polish_expression[i] = temp
        if(not find_consecutive_repeated_letters(polish_expression[i-1:i+3])):
          count = 1
        else:
          temp = polish_expression[i+1]
          polish_expression[i+1] = polish_expression[i]
          polish_expression[i] = temp
  return polish_expression
# m3(polish_expression)

def horizontal_polar(coordinates):
  G = nx.Graph()
  for i in range(0, len(coordinates)):
    G.add_node(coordinates[i][3])
  x_list = np.array(G.nodes())
  for i in range(0, len(x_list)):
    for j in range(0, len(coordinates)):
      if(coordinates[j][3] == x_list[i]):
        G.add_edge(coordinates[j][3]+coordinates[j][1], x_list[i])
  nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')
  plt.show()
horizontal_polar(coordinates1)

def vertical_polar(coordinates):
  G = nx.Graph()
  for i in range(0, len(coordinates)):
    G.add_node(coordinates[i][4])
  y_list = np.array(G.nodes())
  for i in range(0, len(y_list)):
    for j in range(0, len(coordinates)):
      if(coordinates[j][4] == y_list[i]):
        G.add_edge(coordinates[j][4]+coordinates[j][2], y_list[i])
  nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')
  plt.show()
vertical_polar(coordinates1)

def simulated_annealing_greedy(node, coordinates, polish_expression, wire_length_matrix):
  initial_score = cost_floorplan(floorplan_area(coordinates), routing_length(wire_length_matrix, coordinates))
  print(f"Initial score: {initial_score}")
  plot_block(coordinates)
  best_score = initial_score
  for i in range(0, 3):
    coordinates2 = list()
    node1 = None
    temp = []
    x = random.randint(1, 3)
    if(x == 1):
      temp = m1(polish_expression)
      print("Performing move m1")
    elif(x == 2):
      temp = m2(polish_expression)
      print("Performing move m2")
    elif(x == 3):
      temp = m3(polish_expression)
      print("Performing move m3")
    node1 = ptog(temp)
    coordinates2 = plot(node1, coordinates2)
    score = cost_floorplan(floorplan_area(coordinates2), routing_length(wire_length_matrix, coordinates2))
    print(f"Polish expression: {polish_expression}")
    print(f"Area: {floorplan_area(coordinates2)}")
    print(f"Wiring length: {routing_length(wire_length_matrix, coordinates2)}")
    print(f"Score at iteration {i}: {score}")
    if(score-best_score < 0):
      print(f"Score - best score = {score-best_score}")
      best_score = score
      node = node1
      polish_expression = temp
      coordinates = coordinates2
    plot_block(coordinates2)
  return node, coordinates, polish_expression
simulated_annealing_greedy(node, coordinates1, polish_expression, wire_length_matrix)

def simulated_annealing_heuristic(node, coordinates, polish_expression, wire_length_matrix, t):
  initial_score = cost_floorplan(floorplan_area(coordinates), routing_length(wire_length_matrix, coordinates))
  print(f"Initial score: {initial_score}")
  plot_block(coordinates)
  prev_score = initial_score
  for i in range(0, 3):
    coordinates2 = list()
    node1 = None
    temp = []
    x = random.randint(1, 3)
    if(x == 1):
      temp = m1(polish_expression)
      print("Performing move m1")
    elif(x == 2):
      temp = m2(polish_expression)
      print("Performing move m2")
    elif(x == 3):
      temp = m3(polish_expression)
      print("Performing move m3")
    node1 = ptog(temp)
    coordinates2 = plot(node1, coordinates2)
    curr_score = cost_floorplan(floorplan_area(coordinates2), routing_length(wire_length_matrix, coordinates2))
    print(f"Polish expression: {polish_expression}")
    print(f"Area: {floorplan_area(coordinates2)}")
    print(f"Wiring length: {routing_length(wire_length_matrix, coordinates2)}")
    print(f"Score at iteration {i}: {curr_score}")
    print(f"Current score - Previous score = {curr_score-prev_score}")
    if(curr_score-prev_score < 0):
      node = node1
      polish_expression = temp
      coordinates = coordinates2
    else:
      r = random.random()
      if(r < math.exp(-((curr_score-prev_score)/t))):
        node = node1
        polish_expression = temp
        coordinates = coordinates2
    plot_block(coordinates2)
    prev_score = curr_score
    t = 0.6*t
  return node, coordinates, polish_expression
simulated_annealing_heuristic(node, coordinates1, polish_expression, wire_length_matrix, t=150)