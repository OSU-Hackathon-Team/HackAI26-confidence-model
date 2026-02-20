## Confidence Model

> [!WARNING]
> Vibe coded, beware the bugs. I do not know what I'm doing.

A classification model that takes in a sequence of features and outputs a confidence score.

### Input

Sources mediapipe for face and hand landmarks, and uses them to calculate a feature vector.
Has a seperate model using Whisper to for live transcription embeddings, negating the need for a loss of information that the actual transcription would have provided.

### Output

Outputs a confidence score, where 1 is confident and 0 is unconfident.



