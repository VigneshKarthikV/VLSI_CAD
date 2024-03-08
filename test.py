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