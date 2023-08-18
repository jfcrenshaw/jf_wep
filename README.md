# jf_wep

To install with dev dependencies:

1. Create a new environment
2. From the root directory, run `pip install -e ".[dev]"`

Some notes:

When I add the ML algorithm, make sure I make it return the right number of zernikes, using something like this:

```python
dsize = (self.jmax - 3) - zk.size  # type: ignore
if dsize > 0:
    zk = np.pad(zk, (0, dsize), "constant")
elif dsize < 0:
    zk = zk[:dsize]
```
