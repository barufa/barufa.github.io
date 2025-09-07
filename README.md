# StuckInLocalMinima

**A personal blog by Bruno Baruffaldi**  
Senior Machine Learning Engineer | Deep Learning, Computer Vision & High-Performance software.

This repository hosts my personal website and blog, built with [Jekyll](https://jekyllrb.com/) and the [al-folio](https://github.com/alshedivat/al-folio) theme.

---

## 📦 Repository Structure

- `_config.yml` – Global Jekyll and theme configuration.
- `_pages/about.md` – “About” page with my profile and bio.
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

### 2. Install Dependencies

```bash
# Inside the DevContainer
bundle install
npm install  # if you add JS plugins
```

### 3. Start Jekyll

```bash
bundle exec jekyll serve --livereload
```

Your site will be available at http://localhost:4000 with live reloading.

---

## 📄 About the Author

**Bruno Baruffaldi** 👋  
Senior Machine Learning Engineer and software developer.  
Experience in embedded real-time inference (Nvidia Jetson), cloud-scale model serving, and high-performance optimization.  
First engineer at [DeepAgro](https://deepagro.com), building AI systems for targeted herbicide spraying.

**About the Blog**  
“StuckInLocalMinima” is where I share lessons learned, experiments that (mostly) succeeded, and thoughts on ML projects and code.

---

## 📜 License

This project is licensed under the terms described in [`LICENSE`](LICENSE).
