# Why the code you have written for plotting does not work.

Obviously this title is meant to be tongue-in-cheek. The code I left you with was crap. It was bound to break during your efforts to clean it up. 

The original error we got was the following:

`UnboundLocalError: local variable 'ax' referenced before assignment`

on line 72 of `graph.py`. 

A look at this file reveals that `ax` is initialized in the following for-loop:

```
for i, (metric, df) in enumerate(dfs.items()):
        ax = axlist[i]
        ...
```

My thinking is that `dfs` must be empty, if `ax` is never getting initialized. I added the following block:

```
#===DEBUG===
# Sanity check.
assert len(dfs) > 0
assert len(dfs) == len(axlist)
#===DEBUG===
```

Which confirmed my hunch:

```
Traceback (most recent call last):
  File "plotting/plotExperiment.py", line 37, in <module>
    main(**args)
  File "plotting/plotExperiment.py", line 26, in main
    graph(dfs, args['experimentName'], plotFile)
  File "/homes/3/whitaker.213/packages/gcnn/growing-cnns/plotting/graph.py", line 25, in graph
    assert len(dfs) > 0
AssertionError
``` 

The function `graph.graph` is being called inside `plotExperiment.py`, at the end of the `main` function:

```
def main(**kwargs):

    # Read log
    projectRoot = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    logFilename = os.path.join(projectRoot, 'experiments',
            kwargs['experimentName'], '%s.log' % kwargs['experimentName'])
    dfs = read_log(logFilename)


    # Create plot
    plotFile = os.path.join(projectRoot, 'experiments', kwargs['experimentName'],
            '%s.svg' % kwargs['experimentName'])
    graph(dfs, kwargs['experimentName'], plotFile)
``` 

I added the following block to check if the output of `preprocessing.read_log` is empty:

```
    #===DEBUG===
    print(dfs)
    exit()
    #===DEBUG===
```

It was. The `read_log` function is quite a bit different than what I wrote. It appears the optional argument `lengths` isn't passed in `plotExperiment.py`, and so the following block always executes:

```
            if lengths is None:
                #===DEBUG===
                print("lengths is None.")
                #===DEBUG===
                continue
```

And so we never add anything to the `dfs` dict. Long story short, I fixed it by calling your `preprocessing.getLogLengths` function in `plotExperiment.py`. I added some debug statements in the three files mentioned above which will need to be removed. I suggest cleaning them up and merging this branch with master, as I recreated `develop_b` today from whatever your latest commit to master was.  
