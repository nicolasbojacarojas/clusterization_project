# Project Structure

This project is structured to facilitate data analysis and exploration. Below is the directory structure:

- **data/**: Contains all datasets used in the project.
- **dev/**: Development notebooks and scripts.
- **app/**: Dashboards and applications (e.g., Streamlit).
- **docs/**: Documentation for the project.
- **src/**: Source code, scripts, and modules.
- **test/**: Unit and integration tests.
- **main.py**: Main file to execute the project.
- **README.md**: Project description and setup instructions.

## Data Paths

Ensure that all data paths in your scripts and notebooks are updated to reflect the new structure. For example:
- Use `data/gaia_parallax5.csv` to access the dataset instead of the previous relative paths.

## Best Practices

- Keep your code modular and well-documented.
- Avoid hardcoding paths; use relative paths based on the project structure.
- Regularly update the README with any changes to the project structure or dependencies.
