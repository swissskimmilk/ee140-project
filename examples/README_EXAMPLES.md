# Examples Directory

This directory contains reference examples and learning materials that are **not part of the main design flow**.

## Files

### `main.py`
Reference script from Lab 2 containing tutorial exercises for the gm/ID methodology.

**Purpose:** Educational reference showing how to use the lookup tables and design simple amplifier stages.

**Exercises Included:**
- **Exercise 1:** Basic single-stage amplifier design
- **Exercise 2:** Channel length optimization for fT and gain
- **Exercise 3:** Design space exploration with multiple constraints
- **Exercise 4:** Optimizing for maximum gain with load
- **Exercise 5:** Complete differential pair design with current mirror

**Note:** This file is standalone and demonstrates concepts. The actual Lab 2 design is implemented in the main directory scripts (`design_stage1.py`, `design_output_stage.py`, etc.).

## Usage

To run the examples:

```bash
cd examples
python main.py
```

The script will execute Exercise 5 by default (complete differential pair design). You can modify the last line to run other exercises:

```python
# Change from:
exercise_5()

# To run other exercises:
exercise_1()  # Basic amplifier
exercise_2()  # Length optimization
exercise_3()  # Design space exploration  
exercise_4()  # Maximum gain optimization
```

## Relationship to Main Design

The main Lab 2 design uses these concepts but is organized differently:

| Example Concept | Used In Main Design |
|----------------|---------------------|
| gm/ID methodology | `design_stage1.py`, `design_output_stage.py` |
| fT requirements | `calculate_design_params.py` |
| Device sizing | Both stage design scripts |
| Differential pair | `design_stage1.py` (telescopic) |
| Current mirrors | `design_stage1.py` (tail and loads) |

## Learning Path

1. **Start here** if you're new to gm/ID design methodology
2. Read through each exercise to understand the concepts
3. Then look at `design_stage1.py` and `design_output_stage.py` to see how these concepts are applied in the full amplifier design

## References

These examples are based on the gm/ID design methodology taught in EE140/240A. For more information, see:
- Course notes on gm/ID design
- "Systematic Design of Analog CMOS Circuits" by Jespers and Murmann
- Lab 2 handout

