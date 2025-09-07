// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "Publications by categories in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "Repositories",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "Curriculum Vitae",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-from-scikit-learn-to-faiss-migrating-pca-for-scalable-vector-search",
        
          title: "From scikit-learn to Faiss: Migrating PCA for Scalable Vector Search",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/sklearn-faiss/";
          
        },
      },{id: "post-how-to-start-a-machine-learning-project-before-starting-a-machine-learning-project",
        
          title: "How to Start a Machine Learning Project Before Starting a Machine Learning Project...",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/start-ml-project/";
          
        },
      },{id: "post-dvc-many-files-a-strategy-for-efficient-large-dataset-management",
        
          title: "DVC + Many Files: A Strategy for Efficient Large Dataset Management",
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/dvc-fix/";
          
        },
      },{
        id: 'social-discord',
        title: 'Discord',
        section: 'Socials',
        handler: () => {
          window.open("https://discord.com/users/1196821357959315521", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/barufa", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/brunobaruffaldi", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
