<div align='center'>

<h1>Static Site Generator for the EluciDATA project of Sirris</h1>
<p>Static Site Generator for the EluciDATA project, which focuses on the research and providing any company who wishes to intrigue themselves into AI to explore the options</p>


</div>

# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
- [Roadmap](#compass-roadmap)


## :star2: About the Project

### :dart: Features
- Jekyll
- Tailwind
- SEO
- RSS


## :toolbox: Getting Started

### :bangbang: Prerequisites

- This project uses npm as package manager<a href="https://www.npmjs.com/"> Here</a>
- This project uses jekyll, which is based on Ruby<a href="https://www.ruby-lang.org/en/"> Here</a>
```bash
gem install bundler jekyll
```
- This project uses Tailwind as front-end styling<a href="https://tailwindcss.com/docs/installation"> Here</a>
```bash
npm install -D tailwindcss
```


### :gear: Installation

Bundle the jekyll environment (This requires RubyGems)
```bash
bundle install
```
Install script packages
```bash
npm install -g npm-run-all
```
Install tailwind script access
```bash
npm install -g tailwindcss
```


### :running: Run Locally

Clone the project

```bash
https://github.com/EluciDATALab/elucidatalab.github.io.git

```
Go to the project directory
```bash
cd elucidatalab.github.io
```
Install jekyll environment
```bash
bundle install
```
Install tailwindcss
```bash
npm install -D tailwindcss
```
Start the environment locally (http://localhost:4000/elucidata/)
```bash
npm start
```


## :compass: Documentation

### How to create a blog post

Navigate to the _posts folder and create a new markdown file.
This markdown file should have the following naming conventions: YEAR-MONTH-DAY-title.MARKUP (see <a href="[https://www.npmjs.com/](https://jekyllrb.com/docs/posts/)"> Here</a> for more in depth information as of why)
Example: 2024-02-28-my-first-blogpost.markdown

#### Blogpost parameters

Please make sure the markdown file starts and finishes with 3 dashes: '---'

A blog post should at least contain the following:
* Layout: post
* Title: "your blog title here"
* Date: same naming convention as the blogpost name i.e: date:   2024-02-28 16:17:53 +0100
* author: "your author name here"
* image: "pathfile of blogpost image" i.e "src/assets/algorithm-selection.png"
You have to manually add images in the "src" folder before linking to them in your blog post.
* excerpt: "your excerpt here". This should be a small description on what the blog post will be about i.e: "This is a test post to demonstrate adding additional data to a Markdown file for Jekyll."



An example markdown file could look like this:
```
---
layout: post
title: [TITLE]
date: 2024-07-11
author: [AUTHOR]
email: [EMAIL]
image: src/assets/IMAGE.JPG
categories: [Awards, Projects, Papers]
excerpt: [EXCERPT]
```

### How to create a starter kit

Navigate to the _posts folder and go one folder deeper, under the "skits" folder: _posts/skits/YOUR-MARKDOWN-FILE-HERE.markdown


### Starter kit parameters

Please make sure the markdown file starts and finishes with 3 dashes: '---'

A starter kit should countain the following:
* layout: skits
* title: your title here
* date: same as blogpost format
* author: your author here
* categories: same as blogpost format
* image: same as blogpost format
* description: your description here
* excerpt: your excerpt here
* notebook: url to notebook i.e https://collab.research.google.com/yournotebookhere


An example starter kit markdown could look like this:
```
---
layout: skit_detail
title: [TITLE]
image: ../src/assets/[IMAGE]
date:  [DATE]
author: [AUTHOR]
categories:
    - Anomaly detection
description: [DESCRIPTION]
excerpt: " "
notebook: https://collab.research.google.com/yournotebookhere
---
```
