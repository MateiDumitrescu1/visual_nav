The format of the cache is this.

trust_fm = 1 means we should blindly trust the feature matching homography and warped corners.
trust_fm = -1 means we should blindly NOT trust the feature matching homography and warped corners. instead we will blindly trust the inter-frame candidate.

```json
{
    "frames": {
        "0": {
            "best_rotation": 120.0,
            "trust_fm": 1 | -1
        }
        
    }
}
```