from symbols_tree_adapter import adapt_to_solver
import math_recognition.main as mr
from solver import Solver, solver_output_to_str

wolfram_keys_file = open("wolfram_cloud_keys.txt", "r")
wolfram_keys = wolfram_keys_file.read().split('\n')
consumer_key = wolfram_keys[0]
consumer_secret = wolfram_keys[1]
solver = Solver(consumer_key, consumer_secret)
solver.start_session()
mr.prepare_network()


def calculated(filename):
    baselines_tree = mr.math_recognition_image_by_path(filename)
    str_math_r = ''.join(mr.layout_pass_to_list(baselines_tree))
    str_adapted = adapt_to_solver(baselines_tree)

    return {'Распознано': str_math_r,
            'Вычислено': solver_output_to_str(solver.solve(str_adapted))}
