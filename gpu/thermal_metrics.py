"""Score a junction temperature trace: how much it swings, and how much fatigue those swings cause relative to another run."""
import numpy as np


def rainflow(temperature):
    """Count thermal cycles (the standard rainflow method). Returns each cycle's swing [°C] and its count (1, or 0.5 for a half cycle)."""
    change = np.diff(temperature)
    change = change[change != 0]
    keep = np.r_[True, np.sign(change[1:]) != np.sign(change[:-1]), True]
    levels = np.r_[temperature[0], temperature[0] + np.cumsum(change)][keep] if len(change) else temperature[:1]
    swings, counts, stack = [], [], []
    for level in levels:
        stack.append(level)
        while len(stack) >= 3:
            newest = abs(stack[-1] - stack[-2])
            previous = abs(stack[-2] - stack[-3])
            if newest < previous:
                break
            if len(stack) == 3:
                swings.append(previous); counts.append(0.5)
                stack.pop(0)
            else:
                swings.append(previous); counts.append(1.0)
                del stack[-3:-1]
    swings += list(np.abs(np.diff(stack)))
    counts += [0.5] * (len(stack) - 1)
    return np.array(swings), np.array(counts)


def fatigue(temperature, exponent, smallest_swing=0.0):
    """Coffin-Manson: each cycle's damage grows as swing ** exponent. Only meaningful as a ratio between two runs of the same chip. Swings below smallest_swing [°C] count as harmless."""
    swings, counts = rainflow(temperature)
    counted = swings >= smallest_swing
    return float(np.sum(counts[counted] * swings[counted] ** exponent))


def time_std(time, temperature):
    """Standard deviation of the temperature, weighting each point by how long it lasted."""
    weight = np.diff(time)
    value = temperature[:-1]
    mean = np.sum(weight * value) / np.sum(weight)
    return float(np.sqrt(np.sum(weight * (value - mean) ** 2) / np.sum(weight)))
