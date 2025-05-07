import time
import MatrixSummer as MatrixSum
import MatrixGenerator as mg
import numpy as np
from bit_ds import NdBIT as NDBit
from random import randint
import argparse
import json

twoDTests = 10

output = {}


test_dict = {
    "1d_fen": 0,
    "2d_fen": 0,
    "3d_fen": 0,
    "randD_fen": 0
}

def time_scaler(value: float):
    if value < 1000:  # Less than 1 microsecond
        return f"{value}ns"
    elif value < 1_000_000:  # Less than 1 millisecond
        return f"{value/1000:.2f}μs"
    elif value < 1_000_000_000:  # Less than 1 second
        return f"{value/1_000_000:.2f}ms"
    elif value < 60_000_000_000:  # Less than 1 minute
        return f"{value/1_000_000_000:.2f} seconds"
    elif value < 3_600_000_000_000:  # Less than 1 hour
        return f"{value/60_000_000_000:.2f} minutes"
    else:
        return f"{value/3_600_000_000_000:.2f} hours"


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

def generateData(random_range: tuple[int], max_dimension_size: int, min_dimension_size: int, num_matrices: int = 2):
    data = []
    for _ in range(num_matrices):
        data.append(mg.create_random_ndmatrix((randint(min_dimension_size,max_dimension_size), randint(min_dimension_size,max_dimension_size)), random_range))

    # Add more edge cases
    return data

def randomDFenwickSums(dim: tuple[int], queryAmount: int, random_range: tuple[int], verbose: bool, json_output: bool):
    testMatrix = np.array(mg.create_random_ndmatrix(dim, random_range), dtype=int)

    # Create a Fenwick tree from the matrix
    buildTimeStart = time.time_ns()
    fenwick = NDBit(testMatrix, len(dim))
    buildTimeEnd = time.time_ns()
    buildTime = time_scaler(buildTimeEnd - buildTimeStart)

    # Generate random query positions for all tests
    queryPositions =  [[randint(1, dim[i]-1) for i in range(len(dim))] for _ in range(queryAmount)]


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

        queryPosition = queryPositions[test]        
        
        linearStart = time.time_ns()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0 for _ in range(len(dim))], queryPosition)
        linearEnd = time.time_ns()
        lin_times[test] = linearEnd - linearStart
        
        fenwickStart = time.time_ns()
        treeResult = fenwick.sum(queryPosition)
        fenwickEnd = time.time_ns()
        tree_times[test] = fenwickEnd - fenwickStart
        
        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
        
        if verbose > 1:
            print(f"[test {test}] Querying to point: {queryPosition}")
            print(f"[test {test}] Linear: time={time_scaler(lin_times[test])}, result={correct}")
            print(f"[test {test}] FenwickTree: time={time_scaler(tree_times[test])}, result={treeResult}")
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
        print(f"Fenwick Tree build time: {buildTime}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {time_scaler(np.average(lin_times))}")
        print(f"FenwickTree avg: {time_scaler(np.average(tree_times))}")
        print(f"Linear total time: {time_scaler(lin_times.sum())}")
        print(f"Fenwick total time: {time_scaler(tree_times.sum())}")
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

    queryPositions = [[randint(1, dim[0]-1), randint(1, dim[1]-1), randint(1, dim[2]-1)] for _ in range(queryAmount)]

    
    for test in range(queryAmount):
        testMatrix = np.array(mg.create_random_ndmatrix(dim, random_range), dtype=int)

        build_time_start = time.time_ns()
        fenwick = NDBit(testMatrix, len(dim))
        build_time_end = time.time_ns()
        build_time = time_scaler(build_time_end - build_time_start)        


        queryPosition = queryPositions[test]
        
        linearStart = time.time_ns()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0 for _ in range(len(dim))], queryPosition)
        linearEnd = time.time_ns()
        lin_times[test] = linearEnd - linearStart
        
        fenwickStart = time.time_ns()
        treeResult = fenwick.sum(queryPosition)
        fenwickEnd = time.time_ns()
        tree_times[test] = fenwickEnd - fenwickStart
        
        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
        
        if verbose > 1:
            print(f"[test {test}] Querying to point: {queryPosition}")
            print(f"[test {test}] Linear: time={time_scaler(lin_times[test])}, result={correct}")
            print(f"[test {test}] FenwickTree: time={time_scaler(tree_times[test])}, result={treeResult}")
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
        print(f"Fenwick Tree build time: {build_time}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {time_scaler(np.average(lin_times))}")
        print(f"FenwickTree avg: {time_scaler(np.average(tree_times))}")
        print(f"Linear total time: {time_scaler(lin_times.sum())}")
        print(f"Fenwick total time: {time_scaler(tree_times.sum())}")
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
    buildTimeStart = time.time_ns()
    fenwick = NDBit(testArray, 1)
    buildTimeEnd = time.time_ns()
    buildTime = time_scaler(buildTimeEnd - buildTimeStart)

    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)

    queryPositions = [randint(1, matrix_size-1) for _ in range(queryAmount)]
    for i, queryPosition in enumerate(queryPositions):
        
        linearStart = time.time_ns()
        correct = MatrixSum.linear_matrix_sum(testArray, [0], [queryPosition])
        linearEnd = time.time_ns()
        lin_times[i] = linearEnd - linearStart
        treeStart = time.time_ns()
        treeResult = fenwick.sum([queryPosition])
        treeEnd = time.time_ns()
        tree_times[i] = treeEnd - treeStart

        if vervose > 1:
            print(f"[test {i}] Querying to point: {queryPosition}")
            print(f"[test {i}] Linear: time={time_scaler(lin_times[i])}), result={correct}")
            print(f"[test {i}] FenwickTree: time={time_scaler(tree_times[i])}), result={treeResult}")

        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"
    if vervose > 0:
        print(f"Fenwick Tree build time: {buildTime}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {time_scaler(np.average(lin_times))}")
        print(f"FenwickTree avg: {time_scaler(np.average(tree_times))}")
        print(f"Linear total time: {time_scaler(lin_times.sum())}")
        print(f"Fenwick total time: {time_scaler(tree_times.sum())}")
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

    buildTimeStart = time.time_ns()
    fenwick = NDBit(testMatrix, 2)
    buildTimeEnd = time.time_ns()
    buildTime = time_scaler(buildTimeEnd - buildTimeStart)

    lin_times = np.zeros(queryAmount)
    tree_times = np.zeros(queryAmount)

    queryPositions = [[randint(1, MatrixDimensions[0]-1), randint(1,MatrixDimensions[1]-1)] for _ in range(queryAmount)]
    
    for i, queryPosition in enumerate(queryPositions):
        linearStart = time.time_ns()
        correct = MatrixSum.linear_matrix_sum(testMatrix, [0,0], queryPosition)
        linearEnd = time.time_ns()
        lin_times[i] = linearEnd - linearStart
        treeStart = time.time_ns()
        treeResult = fenwick.sum(queryPosition)
        treeEnd = time.time_ns()
        tree_times[i] = treeEnd - treeStart

        if verbose > 1:
            print(f"[test {i}] Querying to point: {queryPosition}")
            print(f"[test {i}] Linear: time={time_scaler(lin_times[i])}, result={correct}")
            print(f"[test {i}] FenwickTree: time={time_scaler(tree_times[i])}, result={treeResult}")


        assert correct == treeResult, f"Assertion failed: correct={correct}, treeResult={treeResult}"

    if verbose > 0:
        print(f"Fenwick Tree build time: {buildTime}")
        print(f"Quering to points: {queryPositions}")
        print(f"Linear avg: {time_scaler(np.average(lin_times))}")
        print(f"FenwickTree avg: {time_scaler(np.average(tree_times))}")
        print(f"Linear total time: {time_scaler(lin_times.sum())}")
        print(f"Fenwick total time: {time_scaler(tree_times.sum())}")
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

    parser.add_argument(
        '-m',
        '--matrix',
        nargs='?',
        type=int,
        help="defines the max size of the different matrices to be used in the tests, it is a list where the first element is the max size of the 2d matrix and the second element is the max size of the 3d matrix and so on",
        default=5000
    )

    parser.add_argument(
        '-s',
        '--min-size',
        nargs='?',
        const=1,
        type=int,
        help="defines the min size of the matrix to be used in the tests",
        default=500
    )

    parser.add_argument(
        '-t',
        '--tests',
        nargs='?',
        const=1,
        type=int,
        help="defines the number of tests to be run",
        default=10
    )

    parser.add_argument(
        '-r',
        '--range',
        nargs=2,
        type=int,
        help="defines the range of the random numbers to be used in the tests",
        default=(-10, 10)
    )

    parser.add_argument(
        '-a',
        '--array',
        nargs='?',
        const=1,
        type=int,
        help="defines the max size of the array to be used in the tests",
        default=5000
    )

    parser.add_argument(
        '-n',
        '--num-matrices',
        nargs='?',
        const=1,
        type=int,
        help="defines the number of matrices to be used in the tests",
        default=2
    )



    args = parser.parse_args()

    #TODO: add fine grain verbose print to only print some tests
    default_matrix_size = 300

    random_range = tuple(args.range)
    max_dimension_size = []
    if args.matrix.__class__ == int:
        args.matrix = [args.matrix]
    for i in args.matrix:
        if i < 1:
            raise ValueError("Matrix size must be greater than 0")
        max_dimension_size.append(i)

    if len(max_dimension_size) < 3:
        for i in range(3 - len(max_dimension_size)):
            max_dimension_size.append(default_matrix_size)
    
    max_dimension_size = tuple(max_dimension_size)

    max_array_size = args.array
    min_dimension_size = args.min_size
    queryAmount = args.tests
    num_matrices =  args.num_matrices

    
    matrices = generateData(random_range, max_dimension_size[0], min_dimension_size, num_matrices)
    for testMatrix in matrices:
        if args.verbose:
            print("Testing 1D Fenwick Tree")
            print("Matrix dimensions:", (max_array_size,))
        oneDFenwickSums(queryAmount, max_array_size, args.verbose, args.json) 

        matrix_dimension = (randint(min_dimension_size,max_dimension_size[0]), randint(min_dimension_size,max_dimension_size[0]))
        if args.verbose:
            print("Testing 2D Fenwick Tree")
            print("Matrix dimensions:", matrix_dimension)
        twoDFenwickSums(queryAmount ,matrix_dimension, args.verbose, args.json)

        nd_dim = [randint(min_dimension_size,max_dimension_size[1]), randint(min_dimension_size,max_dimension_size[1]),randint(min_dimension_size,max_dimension_size[1])]

        if args.verbose:
            print("Testing 3D Fenwick Tree")
            print("Matrix dimensions:", nd_dim)

        threeDFenwickSums(nd_dim, queryAmount, random_range, args.verbose, args.json)

        # Generate a random dimension
        nd_dim = random_dim_generator(max_dimension_size[2])

        if args.verbose:
            print("Testing Random Dimension Fenwick Tree")
            print("Matrix dimensions:", list(nd_dim))
        randomDFenwickSums(nd_dim, queryAmount, random_range, args.verbose, args.json)


    if args.json:
        with open("output.json", "w") as f:
            json.dump(output, f, indent=4)