# Generative Models and Research Analysis

This directory contains documents that provides:

- **Literature review** on generative models from various articles and journals.
- **Challenges** that might appear while implementing.
- **Insights into popular models**

##### This document is intended to help us understand key topics related to generative models, their research landscape, and potential challenges which we could faced.


* Also explores features critical for generative models in bird sound evaluation, such as:

## [Key Features for BirdSet Evaluation](analysis/which features where used in the BirdSet evaluation.pdf)

- **Audio Recordings**: The primary data source for learning the characteristics of bird sounds.
- **Frequency Range (Low Frequency and High Frequency)**: Defines the range of frequencies for synthesized bird sounds.
- **Species Identification (eBird Codes)**: Helps condition the generative model to produce bird sounds specific to certain species.
- **Call Type**: Useful for generating different bird vocalization types (songs, calls, alarms).
- **Start and End Time**: Critical for creating audio segments of appropriate length and structure.
- **Detected Events and Event Clusters**: Provides patterns and structures for replication in generated sounds.
- **Recording Quality**: Ensures clarity and fidelity in generated audio.

Additional relevant metadata:
- **Geographical Location (Lat/Long)**: Adds context to the recordings.
- **Recording Length**: Helps in understanding the complete vocalization.
- **Microphone Type**: Affects audio quality and frequency response.
- **Recorder Info**: Helps understand recording conditions and possible biases.
- **Environmental Factors and Behavioral Characteristics**: Useful for broader analysis, such as migration patterns, diet, or human impact.

For **acoustic analysis**:
- The primary focus should be on **audio characteristics** like frequency range, time intervals, and species identification, while metadata like geographical location and quality can enhance the analysis.
