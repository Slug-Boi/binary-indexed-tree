# Binary-Indexed-Tree (Fenwick Tree)

[![PyPI version](https://img.shields.io/pypi/v/bit_ds.svg)](https://pypi.org/project/bit-ds/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **high-performance** Rust (via PyO3) implementation of a **Binary Indexed Tree** (Fenwick Tree), designed for efficient prefix sum queries and point updates in Python.

## Features
- **O(log n)** point updates and prefix sum queries.
- **0-based indexing** (for intuitive Python compatibility).
- Supports **negative values** in the tree.
- Rust backend for **blazing-fast** operations.

## Why Use a Fenwick Tree?
Fenwick Trees are ideal for problems requiring **dynamic prefix sums** or **frequent updates**, where a static array would be too slow. Common use cases include:
1. **Real-time prefix sums**:  
   - Compute sums over arbitrary ranges in logarithmic time (e.g., financial tracking, scoring systems).
2. **Coordinate compression in algorithms**:  
   - Used in competitive programming for problems like inversion counting or range updates.
3. **Efficient updates**:  
   - Unlike prefix sum arrays (which require O(n) updates), Fenwick Trees handle updates in O(log n).
4. **Low memory overhead**:  
   - Uses only O(n) space, unlike segment trees which require O(4n).

### Comparison to Alternatives
| Structure               | Build Time | Update Time | Prefix Sum Time | Space      | Use Case                     |
|-------------------------|------------|-------------|------------------|------------|------------------------------|
| **Binary Indexed Tree** | O(n log n)       | O(log n)    | O(log n)         | O(n)       | Dynamic prefix sums           |
| Prefix Sum Array        | O(n)       | O(n)        | O(1)             | O(n)       | Static arrays (no updates)    |
| Segment Tree            | O(n)       | O(log n)    | O(log n)         | O(4n)      | Flexible range operations but heavier    |


```bash
pip install bit_ds
```

## Usage
There are 2 classes defined by the library:
1. `BIT`: A class for creating a Binary Indexed Tree (Fenwick Tree) using a list of integers.
2. `NdBIT`: A class for creating a multi-dimensional Binary Indexed Tree (Fenwick Tree) using a numpy list of integers.

### Requirements
- Python 3.12+

**Optional:**
- `numpy` (for NdBIT)
- Rust 1.70+ (for building from source)


### Basic Operations (0-based Indexing)
```python
from bit_ds import BIT

# Initialize a BIT using a list of integers
bit = BIT([1, 2, 3, 4, 5])

# Point update: Add 5 to index 2 (0-based)
bit.update(2, 5)

# Prefix sum: Sum from index 0 to 2 (inclusive, 0-based) 
print(bit.query(2))  # Output: 6 

# Range sum: Sum from index 2 to 4 (inclusive, 0-based)
print(bit.range_query(2, 4))  # Output: 12
```

## API Reference (0-based Indexing)
### `BIT(input: list)`
- Creates a new BIT instance using an input list of integers.
### Methods
| Method                | Description                          | Time Complexity |
|-----------------------|--------------------------------------|-----------------|
| `update(index: int, delta: int)` | Updates the value at `index` (0-based) by `delta`. | O(log n) |
| `query(index: int) -> int`    | Returns the sum from `[0, index]` (inclusive). | O(log n) |
| `range_query(l: int, r: int) -> int` | Returns the sum from `[l, r]` (inclusive). | O(log n) |

### Key Notes
1. **0-based Indexing**:  
   - `update(0, x)` affects the **first element**.  
   - `query(3)` returns the sum from the **first** to the **fourth** element.  
2. **Range Queries**:  
   - `range_query(l, r)` is equivalent to `query(r) - query(l-1)` (handles bounds automatically).  
3. **Negative Deltas**:  
   - Use `update(i, -5)` to subtract values.



## Benchmarks
*WIP*

## Contributing
Pull requests welcome! For major changes, open an issue first.

## License
[MIT](https://choosealicense.com/licenses/mit/)

