# Usage

The evaluation of proposals can be carried out by calling `evaluate_proposals` defined in the `semkittieval.py` script, by setting the path to proposals and dataset.

```python
import semkittieval
# set path to proposals
proposals_path = '/path/to/proposals'
# set path to semantic-kitti
semantic_kitti_base = '/path/to/semantic-kitti'
# evaluate, returns a dict
eval_results = semkittieval.evaluate_proposals(proposals_path, semantic_kitti_base, nthreads=4, nproposals=400, split='test')
```

The evaluation result is a dict-object, whose keys are the object classes like car, person, etc., as well as an additional "mean", which means the mean average recall of all objects. The values of this dict object are numpy arrays, which record how the recall change when number of proposal grows. For example, `eval_results['car'][99]` means the average recall of cars when 100 objects are accepted. 

In `main.py`, a table of average recalls is generated from the `eval_results` and then printed to `stdout`, besides. Additionally, a plot of recall vs. number of proposals is created with `matplotlib`.
