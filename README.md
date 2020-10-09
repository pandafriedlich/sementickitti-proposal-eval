# Usage
```python
import semkittieval
# set path to proposals
proposals_path = '/path/to/proposals'
# set path to semantic-kitti
semantic_kitti_base = '/path/to/semantic-kitti'
# evaluate, returns a dict
eval_results = semkittieval.evaluate_proposals(proposals_path, semantic_kitti_base, nthreads=4, nproposals=400, split='train')
```
