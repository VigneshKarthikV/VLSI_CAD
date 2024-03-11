import unittest
import assignment2

size_of_blocks = assignment2.read_text_file('blocks.txt')

class TestClass(unittest.TestCase):
    def test_read_polish_expression(self):
        self.assertEqual(assignment2.read_polish_expression('polish_expression.txt'), [1, 2, 'V', 3, 'H', 4, 5, 6, 'V', 'H', 7, 'V', 'H'])
        self.assertRaises(ValueError, assignment2.read_polish_expression, 'polish_expression_invalid.txt')
    def test_ptog(self):
        self.assertEqual(assignment2.ptog(assignment2.read_polish_expression('polish_expression.txt')).data, assignment2.read_polish_expression('polish_expression.txt')[-1])
    def test_plot(self):
        coordinates = list()
        coordinates = assignment2.plot(assignment2.ptog(assignment2.read_polish_expression('polish_expression.txt')), coordinates)
        for i in range(0, len(coordinates)):
            self.assertEqual(coordinates[i][1], size_of_blocks[i][1])
            self.assertEqual(coordinates[i][2], size_of_blocks[i][2])
    def test_plot1(self):
        node = assignment2.ptog(assignment2.read_polish_expression('polish_expression.txt'))
        coordinates = list()
        coordinates = assignment2.plot(assignment2.ptog(assignment2.read_polish_expression('polish_expression.txt')), coordinates)
        if(node.left != None):
            node = node.left
            for i in range(0, len(coordinates)):
                if(node.data == coordinates[i][0]):
                    self.assertEqual(node.xl, coordinates[i][3])
                    self.assertEqual(node.yl, coordinates[i][3])
                    self.assertEqual(node.xr, coordinates[i][3] + coordinates[i][1])
                    self.assertEqual(node.yr, coordinates[i][3] + coordinates[i][2])

if __name__ == '__main__':
    unittest.main()