from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.api import init_app
import librosa
import numpy as np

class VolumeClassifier(LabelStudioMLBase):
    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            audio_url = task['data'].get('audio')
            audio_path = self.get_local_path(audio_url)

            try:
                y, sr = librosa.load(audio_path, sr=None)
                frame_length = int(0.001 * sr)
                hop_length = frame_length

                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
                result = []

                for i, energy in enumerate(rms):
                    db = 20 * np.log10(max(energy, 1e-10))
                    if db < -35:
                        continue
                    label = '코골이' if db < -25 else '심한 코골이'
                    result.append({
                        "from_name": "tag",
                        "to_name": "audio",
                        "type": "labels",
                        "value": {
                            "start": float(times[i]),
                            "end": float(times[i] + 0.001),
                            "labels": [label]
                        }
                    })

                predictions.append({
                    "result": result,
                    "score": 0.9
                })

            except Exception as e:
                print(f"오류: {e}")

        return predictions

if __name__ == "__main__":
    app = init_app(VolumeClassifier)
    app.run(host="0.0.0.0", port=9090, debug=True)
