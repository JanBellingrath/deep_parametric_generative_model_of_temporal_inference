#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- utilities -----------------------------
def softmax(X):
    norm = np.sum(np.exp(X) + 1e-5)
    return (np.exp(X) + 1e-5) / norm

def softmax_dim2(X):
    norm = np.sum(np.exp(X) + 1e-5, axis=0)
    return (np.exp(X) + 1e-5) / norm

# ----------------------------- model setup -----------------------------
T = 100
Pi2 = np.zeros((2, T))  # policy posterior over attentional actions (stay, switch)

E2 = np.array([0.99, 0.99])
gammaG2 = 4.0
C2 = np.array([0.99, 0.01])  # prob-space preferences (we will use log(C2))

X2 = np.zeros((2, T))     # attention prior beliefs
X2bar = np.zeros((2, T))  # attention posterior beliefs

x2 = np.zeros(T)          # generative process attentional state (drives true precision)
x2[0] = 0
u2 = np.zeros(T)

X1 = np.zeros((2, T))
X1bar = np.zeros((2, T))

O = np.zeros(T)
O[int(T/5)]   = 1
O[int(2*T/5)] = 1
O[int(3*T/5)] = 1
O[int(4*T/5)] = 1

O1 = np.zeros((2, T))
O1bar = np.zeros((2, T))
for t in range(T):
    O1bar[int(O[t]), t] = 1

X2[:, 0] = [0.5, 0.5]
X1[:, 0] = [0.5, 0.5]

# transitions
B2a = np.zeros((2, 2))
B2b = np.zeros((2, 2))
B2a[:, 0] = [0.8, 0.2]
B2a[:, 1] = [0.0, 1.0]
B2b[:, 0] = [0.0, 1.0]
B2b[:, 1] = [1.0, 0.0]

B1 = np.zeros((2, 2))
B1[:, 0] = [0.8, 0.2]
B1[:, 1] = [0.2, 0.8]

# likelihoods
A1 = np.zeros((2, 2))
A1[:, 0] = [0.75, 0.25]
A1[:, 1] = [0.25, 0.75]
gammaA1 = np.zeros(T)

betaA1m = np.array([0.5, 2.0])  # inverse precisions associated with focused/distracted

A2 = np.zeros((2, 2))
A2[:, 0] = [0.75, 0.25]
A2[:, 1] = [0.25, 0.75]
gammaA2 = 1.0
A2 = softmax_dim2(np.log(A2) * gammaA2)

H2 = np.zeros(2)
H2[0] = np.inner(A2, np.log(A2))[0, 0]
H2[1] = np.inner(A2, np.log(A2))[1, 1]

# trackers
num_policies = 2
action_tracker = []          # 0=no action selection, 1=congruency, 2=incongruency
policy_tracker = []          # just for debugging
policy_EFEs = {0: np.nan, 1: np.nan}          
EFEs_last_epoch = {0: float("inf"), 1: float("inf")}

risk_over_time = np.full(T, np.nan)

threshold = 2.2
skip_counter = 0

last_selected_t = None

# ----------------------------- simulation -----------------------------
for t in range(T - 2):

    betaA1 = np.sum(betaA1m * np.inner(A2, X2[:, t]))  # Bayesian model average over inverse precision
    gammaA1[t] = betaA1m[int(x2[t])] ** -1             # true precision (generative process)

    A1bar = softmax_dim2(A1 ** gammaA1[t])             # precision-weighted likelihood
    O1[:, t] = np.inner(A1bar, X1[:, t])               # observation prior
    X1bar[:, t] = softmax(np.log(X1[:, t]) + gammaA1[t] * np.log(A1[int(O[t]), :]))

    # affective/attentional charge (epsilon)
    AtC = 0.0
    for i in range(2):
        for j in range(2):
            AtC += (O1bar[i, t] - A1bar[i, j]) * X1bar[j, t] * np.log(A1[i, j])

    if AtC > betaA1m[0]:
        AtC = betaA1m[0] - 1e-5

    betaA1bar = betaA1 - AtC

    # attentional state posterior via BMR evidence term
    X2bar[:, t] = softmax(
        np.log(X2[:, t]) - np.log((betaA1m - AtC) / betaA1m * betaA1 / betaA1bar)
    )

    # ------------------ resource-sensitive control ------------------
    if skip_counter > 0:
        skip_counter -= 1

        if last_selected_t is None:
            # should not happen in practice, but guard anyway
            B2 = B2a
            u2[t] = 0
        else:
            B2 = B2a * Pi2[0, last_selected_t] + B2b * Pi2[1, last_selected_t]
            u2[t] = int(np.argmax(Pi2[:, last_selected_t]))

        X2[:, t + 1] = np.inner(B2, X2bar[:, t])
        X1[:, t + 1] = np.inner(B1, X1bar[:, t])

        if u2[t] == 0:
            x2[t + 1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])])
        else:
            x2[t + 1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])])

        action_tracker.append(0)
        policy_tracker.append(2)
        continue

    # else: skip_counter == 0 -> policy simulation takes place
    current_policy = (t // 2 + t) % num_policies  # alternating order across epochs

    if current_policy == 0:
        X2a = np.inner(B2a, X2bar[:, t])
        O2a = np.inner(A2, X2a)
        EFE = np.sum(O2a * (np.log(O2a) - np.log(C2)) - X2a * H2)
        Risk = np.sum(O2a * (np.log(O2a) - np.log(C2)))
    else:
        X2b = np.inner(B2b, X2bar[:, t])
        O2b = np.inner(A2, X2b)
        EFE = np.sum(O2b * (np.log(O2b) - np.log(C2)) - X2b * H2)
        Risk = np.sum(O2b * (np.log(O2b) - np.log(C2)))

    policy_EFEs[current_policy] = float(EFE)
    risk_over_time[t] = float(Risk)

    if (
        t > 1
        and (t % 2 == 0)
        and (EFE < EFEs_last_epoch[0])
        and (EFE < EFEs_last_epoch[1])
        and (EFE < threshold)
    ):
        immediate_selection = np.zeros(num_policies)
        immediate_selection[current_policy] = 1.0
        Pi2[:, t] = immediate_selection
        last_selected_t = t

        B2 = B2a * Pi2[0, t] + B2b * Pi2[1, t]
        X2[:, t + 1] = np.inner(B2, X2bar[:, t])
        X1[:, t + 1] = np.inner(B1, X1bar[:, t])

        u2[t] = int(np.argmax(Pi2[:, t]))
        if u2[t] == 0:
            x2[t + 1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])])
        else:
            x2[t + 1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])])

        action_tracker.append(1)
        skip_counter = 1  # skip the other policy in this epoch
        policy_tracker.append(current_policy)

        # reset / update "memory"
        EFEs_last_epoch = {0: float("inf"), 1: float("inf")}
        EFEs_last_epoch[current_policy] = float(EFE)
        continue

    # first policy of epoch but not immediately selected: propagate using last selected policy (or stay at t==0)
    if t == 0 or (t % 2 == 0):
        if t == 0:
            X2[:, t + 1] = X2[:, t]
            X1[:, t + 1] = np.inner(B1, X1bar[:, t])
            u2[t] = 0
        else:
            # reuse last selected policy distribution
            if last_selected_t is None:
                B2 = B2a
                u2[t] = 0
            else:
                B2 = B2a * Pi2[0, last_selected_t] + B2b * Pi2[1, last_selected_t]
                u2[t] = int(np.argmax(Pi2[:, last_selected_t]))

            X2[:, t + 1] = np.inner(B2, X2bar[:, t])
            X1[:, t + 1] = np.inner(B1, X1bar[:, t])

        if u2[t] == 0:
            x2[t + 1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])])
        else:
            x2[t + 1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])])

        action_tracker.append(0)

    # second policy of epoch: compute Pi2 from both EFEs and enact
    if t % 2 == 1:
        efe_vec = np.array([policy_EFEs[0], policy_EFEs[1]], dtype=float)
        Pi2[:, t] = softmax(np.log(E2) - gammaG2 * efe_vec)
        last_selected_t = t

        B2 = B2a * Pi2[0, t] + B2b * Pi2[1, t]
        X2[:, t + 1] = np.inner(B2, X2bar[:, t])
        X1[:, t + 1] = np.inner(B1, X1bar[:, t])

        selected = int(np.argmax(Pi2[:, t]))
        u2[t] = selected
        if selected == 0:
            x2[t + 1] = np.random.choice([0, 1], p=B2a[:, int(x2[t])])
        else:
            x2[t + 1] = np.random.choice([0, 1], p=B2b[:, int(x2[t])])
        action_tracker.append(1 if selected == current_policy else 2)

        EFE_selected = float(policy_EFEs[selected])
        skip_counter = 2 if (EFE_selected < threshold) else 0

    policy_tracker.append(current_policy)

    # update epoch memory
    EFEs_last_epoch[current_policy] = float(EFE)

print("action_tracker unique values and counts:", np.unique(action_tracker, return_counts=True))

# ============================================================================================================
# Plotting results
##############################################################################################################

import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# -------------------- (1) Track per-time-step policies/actions so the policy strip spans the full timeline ---
time = np.arange(T - 2)

# Deviant times (oddballs)
oddball_ts = np.where(O[:T-2] == 1)[0]

# Risk series (carry forward NaNs, as before)
risk_series = pd.Series(risk_over_time).ffill().iloc[:T-2].to_numpy()

# Build a per-time-step policy-at-time array from policy_tracker:
# 0=stay simulated, 1=switch simulated, 2=no policy simulated (during skipped steps)
policy_at = np.full(T - 2, 2, dtype=int)
n_pol = min(len(policy_tracker), T - 2)
policy_at[:n_pol] = np.array(policy_tracker[:n_pol], dtype=int)

# Build per-time-step action markers from action_tracker.
action_at = np.full(T - 2, 0, dtype=int)
n = min(len(action_tracker), T - 2)
action_at[:n] = np.array(action_tracker[:n], dtype=int)

# -------------------- (2) Palette F4 (Brighter but still Nature-ish) --------------------
PAL = {
    "p1": "#0A3D62",
    "p2": "#14919B",
    "risk": "#2F6690",
    "policy_stay": "#5B7DC7",
    "policy_switch": "#74B3CE",
    "policy_none": "#C8C8C8",
    "congr_none": "#90979D",
    "congr_cong": "#90EE90",
    "congr_incong": "#006400"
}

# -------------------- (3) Figure style --------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.size": 10.8,
    "axes.titlesize": 12.5,
    "axes.labelsize": 10.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
})

fig = plt.figure(figsize=(10, 8), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 3.0])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

# Panel letters
for ax, letter in zip([ax1, ax2, ax3], "ABC"):
    ax.text(-0.06, 1.03, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold")

def add_deviant_shading(ax, alpha=0.10):
    for t0 in oddball_ts:
        ax.axvspan(t0 - 0.5, t0 + 0.5, color="0.5", alpha=alpha, lw=0, zorder=0)

# -------------------- (4) Panel A --------------------
add_deviant_shading(ax1, alpha=0.10)
ax1.plot(time, X1bar[0, :T-2], lw=2.4, color=PAL["p1"],
         label=r"$P(\mathrm{standard}\mid o_{1:\tau})$")
ax1.scatter(time, 1 - O[:T-2], s=18, color=PAL["p1"],
            zorder=3, alpha=0.85, label="true state (deviant=1)")

ax1.set_ylim(-0.05, 1.05)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["deviant", "standard"])
ax1.set_title("First level: perceptual inference")
ax1.tick_params(labelbottom=False)

# -------------------- (5) Panel B --------------------
add_deviant_shading(ax2, alpha=0.10)
ax2.plot(time, X2bar[0, :T-2], lw=2.4, color=PAL["p2"],
         label=r"$P(\mathrm{focused}\mid o_{1:\tau})$")
ax2.scatter(time, 1 - x2[:T-2], s=18, color=PAL["p2"],
            zorder=3, alpha=0.85, label="true state (focused=1)")

ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["distracted", "focused"])
ax2.set_title("Second level: attentional inference")
ax2.tick_params(labelbottom=False)

policy_colors = {
    0: PAL["policy_stay"],
    1: PAL["policy_switch"],
    2: PAL["policy_none"]
}

# -------------------- (6) Panel C (merged with old D) --------------------
add_deviant_shading(ax3, alpha=0.10)

# Draw risk line: solid when policy simulated, dashed when not
for t0 in range(len(time) - 1):
    # Use dashed if either endpoint has no policy simulated
    if policy_at[t0] == 2 or policy_at[t0 + 1] == 2:
        ax3.plot([time[t0], time[t0 + 1]], [risk_series[t0], risk_series[t0 + 1]],
                 lw=2.4, color=PAL["risk"], ls=(0, (3, 2)), zorder=5)
    else:
        ax3.plot([time[t0], time[t0 + 1]], [risk_series[t0], risk_series[t0 + 1]],
                 lw=2.4, color=PAL["risk"], ls="-", zorder=5)
ax3.fill_between(time, risk_series, alpha=0.08, color=PAL["risk"])

ax3.set_title("Subjective temporal extension")
ax3.set_xlabel(r"time ($\tau$)")
ax3.set_ylabel("self-simulational dissimilarity")
ax3.set_xlim(-0.5, T - 2.5)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

# Calculate y position for policy strip (below the data)
ymin_risk = np.nanmin(risk_series)
yrng = (np.nanmax(risk_series) - np.nanmin(risk_series) + 1e-9)
y_policy_strip = -0.15 * yrng  # position below zero for policy squares

congr_colors = {
    0: PAL["congr_none"],
    1: PAL["congr_cong"],
    2: PAL["congr_incong"]
}

# Draw policy squares below zero
for t0 in time:
    pol = int(policy_at[t0])
    ax3.scatter(t0, y_policy_strip, s=55, marker="s",
                color=policy_colors[pol],
                alpha=1.0, edgecolor="none",
                clip_on=False, zorder=3)

# Draw congruency dots at y=0
y_strip = 0
for t0 in time:
    act = int(action_at[t0])
    ax3.scatter(t0, y_strip, s=34,
                color=congr_colors[act],
                clip_on=False, zorder=3)

# Compute average risk for congruent and incongruent actions
congruent_mask = (action_at == 1)
incongruent_mask = (action_at == 2)
avg_risk_congruent = np.nanmean(risk_series[congruent_mask]) if congruent_mask.any() else np.nan
avg_risk_incongruent = np.nanmean(risk_series[incongruent_mask]) if incongruent_mask.any() else np.nan

# -------------------- (8) Legends --------------------
deviant_patch = mpatches.Patch(facecolor="0.5", alpha=0.10, label="deviant stimulus")

ax1.legend(handles=[
    Line2D([0],[0], color=PAL["p1"], lw=2.4, label=r"$P(\mathrm{standard}\mid o_{1:\tau})$"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=PAL["p1"], markersize=6,
           label="true state"),
    deviant_patch,
], loc="lower right")

ax2.legend(handles=[
    Line2D([0],[0], color=PAL["p2"], lw=2.4, label=r"$P(\mathrm{focused}\mid o_{1:\tau})$"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=PAL["p2"], markersize=6,
           label="true state"),
    deviant_patch,
], loc="lower right")

ax3.legend(handles=[
    Line2D([0],[0], color=PAL["risk"], lw=2.4, label="temporal extension"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=policy_colors[0], markersize=7, label="stay"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=policy_colors[1], markersize=7, label="switch"),
    Line2D([0],[0], marker="s", color="none", markerfacecolor=policy_colors[2], markersize=7, label="no policy simulated"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=congr_colors[0], markersize=6, label="no action selection"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=congr_colors[1], markersize=6, 
           label=f"congruent (avg: {avg_risk_congruent:.2f})"),
    Line2D([0],[0], marker="o", color="none", markerfacecolor=congr_colors[2], markersize=6, 
           label=f"incongruent (avg: {avg_risk_incongruent:.2f})"),
    deviant_patch,
], loc="lower right")

plt.savefig("deep_parametric_generative_model_output.png", dpi=300, bbox_inches="tight")
plt.savefig("deep_parametric_generative_model_output.pdf", bbox_inches="tight")
print("Saved: deep_parametric_generative_model_output.png and .pdf")
plt.show()

