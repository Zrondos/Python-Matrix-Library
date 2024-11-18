# Python Matrix Library
This is a Python matrix library implemented in C for Intel's AVX 256-bit architecture.

See `src/matrix.c` for relevant code.   
Functions of interest:
- `add_matrix` Lines 251-293
- `mul_matrix` Lines 325-390
- `pow_matrix` Lines 398-451


### Implementation details
- Python Library in C for Intel’s AVX 256-bit architecture to optimize matrix operations such as `add`, `multiply`, `exponentiation`, and `map`
- Used Intel Intrinsics (Advanced Vector Extensions 256-bit registers) to perform Single Instruction, Multiple Data instructions
- Used OpenMP to create compiler directives to perform multiple instructions in parallel threads
- Used multi-threading (SIMD and MIMD), cache blocking, and algorithm optimizations to implement matrix operations up to 800 times faster than the Python implementation

