# StuckInLocalMinima

**A personal blog by Bruno Baruffaldi**
Senior Machine Learning Engineer | Deep Learning, Computer Vision & High-Performance software.

This repository hosts my personal website and blog, built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme.

---

## 📦 Repository Structure

- `_config.yml` – Global Jekyll and theme configuration.
- `_pages/about.md` – "About" page with my profile and bio.
- `_posts/` – Blog posts in Markdown format.
- `_data/socials.yml` – Social media links and icons.
- `_includes/`, `_layouts/`, `_sass/` – Theme components and styles.
- `assets/` – Static images, fonts, and scripts.
- `LICENSE` – Project license (MIT).

---

## 🛠️ Local Development

### 1. Open in DevContainer

This project includes a VS Code DevContainer.
Open the command palette (`Ctrl+Shift+P`) and select **Remote-Containers: Reopen in Container**.

> **Note:** The DevContainer features section is currently commented out due to installation issues. Dependencies must be installed manually after attaching to the container (see step 2).

### 2. Install Dependencies

```bash
# System dependencies (inside the DevContainer)
sudo apt-get update
sudo apt-get install -y imagemagick inotify-tools nodejs npm

# Python dependencies
pip3 install --upgrade nbconvert --break-system-packages

# Ruby gems
bundle install

# Node.js packages (prettier + liquid plugin)
npm install
```

### 3. Check Code Formatting

```bash
npx prettier . --check
# To auto-fix formatting issues:
npx prettier . --write
```

### 4. Start Jekyll

```bash
bundle exec jekyll serve --watch --port=8080 --host=0.0.0.0 --livereload
```

Your site will be available at http://localhost:8080 with live reloading.

Alternatively, use the provided script (requires `inotify-tools`):

```bash
bash bin/entry_point.sh
```

---

## 🚀 CI/CD Workflows

| Workflow           | Trigger         | Description                                 |
| ------------------ | --------------- | ------------------------------------------- |
| `prettier.yml`     | push/PR to main | Checks code formatting with Prettier        |
| `deploy.yml`       | push to main    | Builds and deploys the site to GitHub Pages |
| `broken-links.yml` | scheduled       | Checks for broken links in source files     |
| `axe.yml`          | scheduled       | Accessibility checks on the deployed site   |

---

## 📜 License

This project is licensed under the terms described in [`LICENSE`](LICENSE).
