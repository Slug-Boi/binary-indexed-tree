import time
import MatrixSummer as MatrixSum
import MatrixGenerator as mg
import numpy as np
from bit_ds import NdBIT as NDBit
from random import randint
import argparse
import json

twoDTests = 10

matrix_size = 2000

output = {}


test_dict = {
    "1d_fen": 0,
    "2d_fen": 0,
    "3d_fen": 0,
    "randD_fen": 0
}


def random_dim_generator(dim_size: int):
    # close enough this seems to produce slightly above the max size 
    dim_list = []
    while True:
        if dim_size > 1:
            if dim_size > 10:
                dim = randint(2, 10)
            dim_size = dim_size//dim
        else:
            dim = 0
        if dim > 1:
            dim_list.append(dim)
        else:
            break
        
    return tuple(dim_list)


def current_milli_time():
    return round(time.time() * 1000)

def generateData(random_range: tuple[int], max_dimension_size: int, min_dimension_size: int):
    data = []
    for _ in range(2):
        data.append(mg.create_random_ndmatrix((randint(min_dimension_size,max_dimension_size), randint(min_dimension_size,max_dimension_size)), random_range))

    # Add more edge cases
    return data

def randomDFenwickSums(dim: tuple[int], queryAmount: int, random_range: tuple[int], verbose: bool, json_output: bool):
    testMatrix = np.array(mg.create_random_ndmatrix(dim, random_range), dtype=int)

    # Create a Fenwick tree from the matrix
    buildTimeStart = current_milli_time()
    fenwick = NDBit(testMatrix, len(dim))
    buildTimeEnd = current_milli_time()

    queryPosition = [randint(1, dimension - 1) for dimension in dim]


    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)

    if json_output:
        full_test = {
            "dimension": dim,
            "num_tests": queryAmount,
            "random_range": random_range,
            "tests": {}
        }

    
    for test in range(queryAmount):
        
        linearStart = current_milli_time()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0 for _ in range(len(dim))], queryPosition)
        linearEnd = current_milli_time()
        lin_times[test] = linearEnd - linearStart
        
        fenwickStart = current_milli_time()
        treeResult = fenwick.sum(queryPosition)
        fenwickEnd = current_milli_time()
        tree_times[test] = fenwickEnd - fenwickStart
        
        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
        
        if verbose > 1:
            print(f"[test {test}] Querying to point: {queryPosition}")
            print(f"[test {test}] Linear: time={lin_times[test]}, result={correct}")
            print(f"[test {test}] FenwickTree: time={tree_times[test]}, result={treeResult}")
            print()

        # save individual test results
        if json_output:
            full_test["tests"][test] = {
                "query_position": queryPosition,
                "linear_time": lin_times[test],
                "linear_result": int(correct),
                "fenwick_time": tree_times[test],
                "fenwick_result": treeResult
            }
    
    if verbose > 0:
        print(f"Fenwick Tree build time: {buildTimeEnd - buildTimeStart}")
        print(f"Linear avg: {np.average(lin_times)}")
        print(f"FenwickTree avg: {np.average(tree_times)}")
        print(f"Linear total time: {lin_times.sum()}")
        print(f"Fenwick total time: {tree_times.sum()}")
        print()

    # save run values
    if json_output: 
        full_test["linear_avg"] = np.average(lin_times)
        full_test["fenwick_avg"] = np.average(tree_times)
        full_test["linear_total_time"] = lin_times.sum()
        full_test["fenwick_total_time"] = tree_times.sum()
        output[f"randD_fen {test_dict['randD_fen']}"] = full_test
        test_dict["randD_fen"] += 1



def threeDFenwickSums(dim: tuple[int], queryAmount: int, random_range: tuple[int], verbose: bool, json_output: bool):
    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)


    if json_output:
        full_test = {
            "dimension": dim,
            "num_tests": queryAmount,
            "random_range": random_range,
            "tests": {}
        }

    
    for test in range(queryAmount):
        testMatrix = np.array(mg.create_random_ndmatrix(dim, random_range), dtype=int)
        fenwick = NDBit(testMatrix, len(dim))
        
        queryPosition = [randint(1, dimension - 1) for dimension in dim]
        
        linearStart = current_milli_time()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0 for _ in range(len(dim))], queryPosition)
        linearEnd = current_milli_time()
        lin_times[test] = linearEnd - linearStart
        
        fenwickStart = current_milli_time()
        treeResult = fenwick.sum(queryPosition)
        fenwickEnd = current_milli_time()
        tree_times[test] = fenwickEnd - fenwickStart
        
        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
        
        if verbose > 1:
            print(f"[test {test}] Querying to point: {queryPosition}")
            print(f"[test {test}] Linear: time={lin_times[test]}, result={correct}")
            print(f"[test {test}] FenwickTree: time={tree_times[test]}, result={treeResult}")
            print()

        # save individual test results
        if json_output:
            full_test["tests"][test] = {
                "query_position": queryPosition,
                "linear_time": lin_times[test],
                "linear_result": int(correct),
                "fenwick_time": tree_times[test],
                "fenwick_result": treeResult
            }
    
    if verbose > 0:
        print(f"Linear avg: {np.average(lin_times)}")
        print(f"FenwickTree avg: {np.average(tree_times)}")
        print(f"Linear total time: {lin_times.sum()}")
        print(f"Fenwick total time: {tree_times.sum()}")
        print()

    # save run values
    if json_output: 
        full_test["linear_avg"] = np.average(lin_times)
        full_test["fenwick_avg"] = np.average(tree_times)
        full_test["linear_total_time"] = lin_times.sum()
        full_test["fenwick_total_time"] = tree_times.sum()
        global test_num
        output[f"3d_fen {test_dict['3d_fen']}"] = full_test
        test_dict["3d_fen"] += 1

def oneDFenwickSums(queryAmount, matrix_size: int, vervose: int, json_output: bool):    
    testArray = np.array(mg.create_random_ndmatrix((matrix_size,), (-10, 10)), dtype=int)
    buildTimeStart = current_milli_time()
    fenwick = NDBit(testArray, 1)
    buildTimeEnd = current_milli_time()

    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)

    queryPositions = [randint(1, matrix_size-1) for _ in range(queryAmount)]
    for i, queryPosition in enumerate(queryPositions):
        linearStart = current_milli_time()
        correct = MatrixSum.linear_matrix_sum(testArray, [0], [queryPosition])
        linearEnd = current_milli_time()
        lin_times[i] = linearEnd - linearStart
        treeStart = current_milli_time()
        treeResult = fenwick.sum([queryPosition])
        treeEnd = current_milli_time()
        tree_times[i] = treeEnd - treeStart

        if vervose > 1:
            print(f"[test {i}] Querying to point: {queryPosition}")
            print(f"[test {i}] Linear: time={lin_times[i]}, result={correct}")
            print(f"[test {i}] FenwickTree: time={tree_times[i]}, result={treeResult}")

        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
    if vervose > 0:
        print(f"Fenwick Tree build time: {buildTimeEnd - buildTimeStart}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {np.average(lin_times)}")
        print(f"FenwickTree avg: {np.average(tree_times)}")
        print(f"Linear total time: {lin_times.sum()}")
        print(f"Fenwick total time: {tree_times.sum()}")
        print()

    if json_output:
        full_test = {
            "dimension": [1],
            "num_tests": queryAmount,
            "random_range": (-10, 10),
            "tests": {}
        }
        for i in range(queryAmount):
            full_test["tests"][i] = {
                "query_position": queryPositions,
                "linear_time": lin_times[i],
                "linear_result": int(correct),
                "fenwick_time": tree_times[i],
                "fenwick_result": treeResult
            }
        full_test["linear_avg"] = np.average(lin_times)
        full_test["fenwick_avg"] = np.average(tree_times)
        full_test["linear_total_time"] = lin_times.sum()
        full_test["fenwick_total_time"] = tree_times.sum()
        output[f'1d_fen {test_dict["1d_fen"]}'] = full_test
        test_dict["1d_fen"] += 1

    



def twoDFenwickSums(queryAmount, MatrixDimensions: tuple[int], verbose: int, json_output: bool):
    testMatrix = np.array(mg.create_random_ndmatrix(MatrixDimensions, random_range),dtype=int)

    buildTimeStart = current_milli_time()
    fenwick = NDBit(testMatrix, 2)
    buildTimeEnd = current_milli_time()

    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)

    queryPositions = [[randint(1, MatrixDimensions[0]-1), randint(1,MatrixDimensions[1]-1)] for _ in range(queryAmount)]
    
    for i, queryPosition in enumerate(queryPositions):
        linearStart = current_milli_time()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0,0], queryPosition)
        linearEnd = current_milli_time()
        lin_times[i] = linearEnd - linearStart
        treeStart = current_milli_time()
        treeResult = fenwick.sum(queryPosition)
        treeEnd = current_milli_time()
        tree_times[i] = treeEnd - treeStart

        if verbose > 1:
            print(f"[test {i}] Querying to point: {queryPosition}")
            print(f"[test {i}] Linear: time={lin_times[i]}, result={correct}")
            print(f"[test {i}] FenwickTree: time={tree_times[i]}, result={treeResult}")


        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"

    if verbose > 0:
        print(f"Fenwick Tree build time: {buildTimeEnd - buildTimeStart}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {np.average(lin_times)}")
        print(f"FenwickTree avg: {np.average(tree_times)}")
        print(f"Linear total time: {lin_times.sum()}")
        print(f"Fenwick total time: {tree_times.sum()}")
        print()

    if json_output:
        full_test = {
            "dimension": MatrixDimensions,
            "num_tests": queryAmount,
            "random_range": random_range,
            "tests": {}
        }
        for i in range(queryAmount):
            full_test["tests"][i] = {
                "query_position": queryPositions[i],
                "linear_time": lin_times[i],
                "linear_result": int(correct),
                "fenwick_time": tree_times[i],
                "fenwick_result": treeResult
            }
        full_test["linear_avg"] = np.average(lin_times)
        full_test["fenwick_avg"] = np.average(tree_times)
        full_test["linear_total_time"] = lin_times.sum()
        full_test["fenwick_total_time"] = tree_times.sum()
        output[f'2d_fen {test_dict["2d_fen"]}'] = full_test
        test_dict["2d_fen"] += 1


def nDFenwickSums():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--verbose',
        nargs='?',
        const=1,
        type=int,
        help="defines the level of verbosity of the output",
        default=0
    )

    parser.add_argument(
        '-j',
        '--json',
        action='store_true',
        help="defines if the output should be saved in json format",
        default=False
    )

    args = parser.parse_args()

    #TODO: add fine grain verbose print to only print some tests
    random_range = (-10, 10)
    max_dimension_size = 5000
    min_dimension_size = 500
    
    matrices = generateData(random_range, max_dimension_size, min_dimension_size)
    for testMatrix in matrices:
        matrix_dimension = (randint(min_dimension_size,max_dimension_size), randint(min_dimension_size,max_dimension_size))

        if args.verbose:
            print("Testing 1D Fenwick Tree")
            print("Matrix dimensions:", (matrix_size,))
        oneDFenwickSums(3, matrix_size, args.verbose, args.json) 

        if args.verbose:
            print("Testing 2D Fenwick Tree")
            print("Matrix dimensions:", matrix_dimension)
        twoDFenwickSums(3 ,matrix_dimension, args.verbose, args.json)

        nd_dim = [randint(1,200), randint(1,200),randint(1,200)]

        if args.verbose:
            print("Testing 3D Fenwick Tree")
            print("Matrix dimensions:", nd_dim)

        threeDFenwickSums(nd_dim, 3, random_range, args.verbose, args.json)

        # Generate a random dimension
        nd_dim = random_dim_generator(max_dimension_size)

        if args.verbose:
            print("Testing Random Dimension Fenwick Tree")
            print("Matrix dimensions:", nd_dim)
        randomDFenwickSums(nd_dim, 3, random_range, args.verbose, args.json)


    if args.json:
        with open("output.json", "w") as f:
            json.dump(output, f, indent=4)