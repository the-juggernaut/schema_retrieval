# "MkDocs Publisher" Action

I keep repeating the steps to build and deploy our MkDocs documentation sites to GitHub Pages across different repos. Let's create a reusable composite action to handle this.

Action Name: MkDocs Publisher
Purpose: A simple action to build an MkDocs site and push it to the gh-pages branch. Should be easy to use. Author should be listed as 'DevRel Team'.

Inputs Needed:

python-version: We need to specify the Python version for setting up the environment. Users should be able to choose. Let's make this optional and default it to 3.11. Description: 'The version of Python to set up for building.'
requirements-file: Users might have different requirements files (e.g., requirements.txt, docs/requirements.txt). This input should specify the path. It's required. Description: 'Path to the Python requirements file'.
gh-token: The GitHub token for pushing to gh-pages. This is absolutely required. Description: 'GitHub token for deployment.' Let's add a note that this is usually ${{ secrets.GITHUB_TOKEN }}. We should probably add a deprecation message to this input later if we find a better way, maybe: "Prefer using GITHUB_TOKEN environment variable directly if permissions allow."
Outputs:

The action needs to output the URL where the site was deployed. Let's call this output page-url. Its description should be 'The URL of the deployed GitHub Pages site.' The actual value should come from the deployment step (see below).

How it Runs (Execution):

This will be a composite action (using: composite). Here are the steps involved:

Checkout Code: First, we need the repository code. Use the standard actions/checkout@v4.
Setup Python: Next, set up the Python environment. Use actions/setup-python@v5. We must use the python-version input provided by the user here. Let's ID this step as setup_python.
Install Dependencies: Run a command to install the Python packages. The command is pip install -r ${{ inputs.requirements-file }}. Execute this using the bash shell. Maybe give this step the name "Install Python Packages".
Build Site: Run the command mkdocs build. Use bash for this too.
Deploy to Pages: Use an existing action for the deployment. peaceiris/actions-gh-pages@v3 seems popular. Give this step the ID deploy. We need to configure it:
Set its github_token parameter to the gh-token input we defined earlier.
Set its publish_dir parameter to ./site (which is the default build output dir for MkDocs).
Maybe add a condition (if:): github.ref == 'refs/heads/main' so it only deploys from the main branch.

Remember the page-url output? Its value needs to be linked to the output of the deploy step. So, for the page-url output, the value should be ${{ steps.deploy.outputs.page_url }}.

Branding:
For the marketplace look, let's use the color blue and the book-open icon. Looks professional.