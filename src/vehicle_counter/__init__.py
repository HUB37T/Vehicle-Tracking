"""Video-independent contracts and vehicle crossing logic."""

from vehicle_counter.contracts import CountEvent, CountSummary, RunConfig, TrackObservation
from vehicle_counter.counting import VehicleCounter

__all__ = ["CountEvent", "CountSummary", "RunConfig", "TrackObservation", "VehicleCounter"]
