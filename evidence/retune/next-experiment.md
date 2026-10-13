# Integral timing decision

The proposed carryover of learned holding integral was not implemented or tested. The user explicitly chose to keep resetting I on each new waypoint and remove the 0.5 s wait. The app now requests zero integral delay, so integration starts on the next valid control interval. Repeated identical targets preserve I. The prior delay was an implementation choice, not a hardware requirement.

All 15 completed safe-* trials in this directory used the previous 0.5 s delay. Their gains have not been validated with immediate integration; factory P/D and I=0 remain the defaults. The regression test exercises the UI-to-SDK path at a 4 ms control interval, checks accumulation immediately after each reset, and verifies identical-target preservation.
