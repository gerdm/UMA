# PyPremium

PyPremium is a small mid-term project for the Pension's class at the Universidad Marista. The purpose of the project is to compute the premium that an insurance must charge in order to cover the pension of an invalid.

The premium is computed as follows
![premium](images/pension_premium.png)

Where:

- <img src="images/annuity.png" width=30> is an annuity that starts the payment at the beginning of the year; it pays monthly; and it is certain

- <img src="images/kpz.png" width=30> is the probability that member z reaches age z + k

- <img src="images/bij.png" width=30> is the pension the sons would get is the mother is either alive or dead. In the former case, the livelihood of the mother is taken as input to compute the pension.

The input is a python Dictionary of the form:
```python
{"name1": ["invalid",age, "F"/"M", True],
 "name2": ["spouse", age, "F"/"M", False],
 "name3": ["descendant", age, "F"/"M", False],
  ...,
 "namen": [family_status, age, sex, Invalid? (T/F)]}
```

Two of the main methods for the class ``PensionPremium`` are:

```python
PensionPremium.compute_premium()
PensionPremium.steps_table()
```

The former computes the premmium *as is*, while the latter outputs a .csv file with the steps followed inbetween. 

## An example
The following family
```python
 family = {"X": ["invalid", 50, "M", True],
              "Y": ["spouse", 45, "F", False],
              "x1": ["descendant", 20, "M", False],
              "x2": ["descendant", 10, "F", False]}
```

Has a pension premium of 953.993950495, and the first 5 steps looks:



| Y | Px_inv      | Px_spouse   | b1(i)       | b2(i)       | Vk          | P(0) | P(1)        | P(2)        |
|---|-------------|-------------|-------------|-------------|-------------|------|-------------|-------------|
| 0 | 1           | 1           | 4050        | 3600        | 1           | 0    | 0           | 1           |
| 1 | 0.98145     | 0.99851     | 4049.811    | 3599.811    | 0.966183575 | 0    | 0.00063     | 0.99937     |
| 2 | 0.962684676 | 0.996852473 | 4049.60413  | 3599.60413  | 0.9335107   | 0    | 0.001319565 | 0.998680435 |
| 3 | 0.943719788 | 0.995008296 | 4049.376431 | 3599.376431 | 0.901942706 | 0    | 0.002078562 | 0.997921438 |
| 4 | 0.924562276 | 0.992958579 | 4049.127949 | 3599.127949 | 0.871442228 | 0    | 0.002906837 | 0.997093163 |
| 5 | 0.905211188 | 0.990684704 | 3750        | 3300        | 0.841973167 | 0    | 1           | 0           |