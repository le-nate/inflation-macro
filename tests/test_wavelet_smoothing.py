"""Test data smoothing functions"""

# %%
from scripts import wavelet_smoothing as ws

# %%
print("Testing trim_signal catches odd-numbered signals")
test_signal = [x for x in range(1000)]
print(f"Signal length: {len(test_signal)}")
trim = ws.trim_signal(test_signal)
assert len(trim) == len(test_signal)
print(f"Signal length: {len(test_signal)}")
test_signal = [x for x in range(1001)]
print(f"Signal length: {len(test_signal)}")
trim = ws.trim_signal(test_signal)
assert len(trim) != len(test_signal)


# %%
print("Testing smooth signal")

# %%
print("Testing smooth_signal with odd-numbered observations")

print("Wavelet smoothing testing complete.")
