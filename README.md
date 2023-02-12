# Scientific Research with Python Playground

This is a demo repository for doing scientific research with python. In this repo, you can:

- See instructions on how to do scientific research with python;
- Play around with the repo, and learn how to do python;
- Understand how to collaborate with others on a project.

This repo belongs to APRILab.

## Where to start?

For APRILab members, please go through the corresponding seminar lectures to learn how to use this repo. A list of seminars can be found in our [wiki](https://insarlab.atlassian.net/wiki/spaces/~854136102/pages/263782626/Seminar+List).

## Github Collaboration Workflow

**PLEASE FOLLOW THESE STEPS WHEN COLLABORATING ON GITHUB REPOS.**

### 1. Creating a Fork

Go to the GitHub repo you want to collaborate on, and click the "Fork" button.

```shell
# Clone your fork to your local machine / server
git clone git@github.com:yuxiao-qin/scientific_research_with_python_demo.git
```

### 2. Link your forked repo to the upstream

To make sure your fork is up to date with the original "upstream" repo that you forked, do:

```shell
# Add 'upstream' repo to list of remotes
git remote add upstream https://github.com/UPSTREAM-USER/ORIGINAL-PROJECT.git
```

Update your branch to sync with the upstream:

```shell
# Fetch from upstream remote
git fetch upstream
```

Now, checkout your own branch and merge the upstream repo's branch. **Always checkout from the develop branch, merge to develop branch and submit a pull request to the upsteam develop branch.**

```shell
# Checkout your develop branch and merge upstream
git checkout develop
git merge upstream/develop
```

If there are no unique commits on the local master branch, git will simply perform a fast-forward. *However*, if you have been making changes on develop (in the vast majority of cases you **shouldn't** be - [see the next section](#doing-your-work), you may have to deal with conflicts. When doing so, be careful to respect the changes made upstream.

Now, your local develop branch is up-to-date with everything modified upstream.

### Doing Your Dev Work

#### Create Your Own Dev Branch

Whenever you begin work on a new feature or bugfix, it's important that you create a new branch. Not only is it proper git workflow, but it also keeps your changes organized and separated from the develop branch so that you can easily submit and manage multiple pull requests for every task you complete.

To create a new branch and start working on it:

```shell
# Checkout the develop branch - you want your new branch to come from the latest develop branch
git checkout develop

# Create a new branch named newfeature (give your branch its own simple informative name)
git branch -b newfeature
```

Now, go to do your hacking work! Remember to commit often!

```shell
# commit your changes in git
git commit -a -m "Added a new feature that does X"

# push your commits to your fork repo
git push origin newfeature
```

### Submitting a Pull Request

When you feel that your work is ready to be reviewed and merged into the upstream repo, you should submit a pull request. This should be done in the Github Web UI:


### Cleaning Up Your Work

Prior to submitting your pull request, you might want to do a few things to clean up your branch and make it as simple as possible for the original repo's maintainer to test, accept, and merge your work.

If any commits have been made to the upstream master branch, you should rebase your development branch so that merging it will be a simple fast-forward that won't require any conflict resolution work.

```shell
# Fetch upstream master and merge with your repo's master branch
git fetch upstream
git checkout master
git merge upstream/master

# If there were any new commits, rebase your development branch
git checkout newfeature
git rebase master
```

Now, it may be desirable to squash some of your smaller commits down into a small number of larger more cohesive commits. You can do this with an interactive rebase:

```shell
# Rebase all commits on your development branch
git checkout
git rebase -i master
```

This will open up a text editor where you can specify which commits to squash.

### Submitting

Once you've committed and pushed all of your changes to GitHub, go to the page for your fork on GitHub, select your development branch, and click the pull request button. If you need to make any adjustments to your pull request, just push the updates to GitHub. Your pull request will automatically track the changes on your development branch and update.

## Accepting and Merging a Pull Request

Take note that unlike the previous sections which were written from the perspective of someone that created a fork and generated a pull request, this section is written from the perspective of the original repository owner who is handling an incoming pull request. Thus, where the "forker" was referring to the original repository as `upstream`, we're now looking at it as the owner of that original repository and the standard `origin` remote.

### Checking Out and Testing Pull Requests
Open up the `.git/config` file and add a new line under `[remote "origin"]`:

```
fetch = +refs/pull/*/head:refs/pull/origin/*
```

Now you can fetch and checkout any pull request so that you can test them:

```shell
# Fetch all pull request branches
git fetch origin

# Checkout out a given pull request branch based on its number
git checkout -b 999 pull/origin/999
```

Keep in mind that these branches will be read only and you won't be able to push any changes.

### Automatically Merging a Pull Request
In cases where the merge would be a simple fast-forward, you can automatically do the merge by just clicking the button on the pull request page on GitHub.

### Manually Merging a Pull Request
To do the merge manually, you'll need to checkout the target branch in the source repo, pull directly from the fork, and then merge and push.

```shell
# Checkout the branch you're merging to in the target repo
git checkout master

# Pull the development branch from the fork repo where the pull request development was done.
git pull https://github.com/forkuser/forkedrepo.git newfeature

# Merge the development branch
git merge newfeature

# Push master with the new feature merged into it
git push origin master
```

Now that you're done with the development branch, you're free to delete it.

```shell
git branch -d newfeature
```



**Copyright**

Copyright 2017, Chase Pettit

MIT License, http://www.opensource.org/licenses/mit-license.php

**Additional Reading**
* [Atlassian - Merging vs. Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

**Sources**
* [GitHub - Fork a Repo](https://help.github.com/articles/fork-a-repo)
* [GitHub - Syncing a Fork](https://help.github.com/articles/syncing-a-fork)
* [GitHub - Checking Out a Pull Request](https://help.github.com/articles/checking-out-pull-requests-locally)