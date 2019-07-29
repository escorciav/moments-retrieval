Welcome to the garage!!!

Code here is mainly in Jupyter-notebooks and illustrative small steps or the roadmap of our journey.

The code here:
- needs refactoring hence not expect maintainance. Also ask before submitting a contribution.

- was used once or few times only to dump referenced data snapshots.

- The good news, you might find comments and figures to guide you through our journey 😅.

If cells takes a lot of time, we move them onto "/scripts" an let them running on clusters.

__Notes__:


- Why notebooks?

    Research projects are highly non-linear, thus designing good code is quite an overhead for a single person. Plus dealing with data just adds more fire to the non-trivial amount of under specification. Thus, @escorciav ended up setting a middle point, the garage. Here, we just prototype, debug and play while the path becomes clear.

- Some notebooks may not run and that's fine, right?

    As long as you have the correct development environment, errors are trivial to fix. Those related to missing funtions or modules, just need the project folder into the PYTHONPATH.

    How to do it?

    ```python
    import sys
    sys.path.append('..')
    ```

    I trust that you want it so badly that you will make it work. Again, any pull request to files in this folders will be closed as there is no intention to maintain it.

    _Why something that does not work is fine?_

    If I take your notes I may have a hard time navigating through them 😁. We are just releasing them for transparency and opening the discussion if we should keep doing this.