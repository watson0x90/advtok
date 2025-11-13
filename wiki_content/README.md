# AdvTok Research Wiki Content

This directory contains pre-written wiki pages for the AdvTok Research repository.

## How to Use These Pages

### Option 1: Create Wiki Pages Manually (Recommended)

1. Go to your repository wiki: https://github.com/watson0x90/advtok/wiki

2. Create the first page (Home):
   - Click "Create the first page"
   - Copy content from `Home.md`
   - Paste and save

3. Create additional pages:
   - Click "New Page"
   - Use the filename (without .md) as the page title
   - Copy content from the corresponding .md file
   - Save

### Option 2: Clone and Push Wiki Repository

Once you've created the first wiki page manually:

```bash
# Clone the wiki repository
git clone https://github.com/watson0x90/advtok.wiki.git
cd advtok.wiki

# Copy wiki content files
cp ../wiki_content/*.md .

# Remove this README (not a wiki page)
rm README.md

# Commit and push
git add *.md
git commit -m "Add comprehensive wiki documentation"
git push origin master
```

## Wiki Pages Included

### Core Pages (Must Create)

1. **Home.md** - Wiki homepage with navigation
2. **Installation-Guide.md** - Complete installation instructions
3. **Quick-Start.md** - 5-minute getting started guide
4. **RTX-5080-Setup.md** - High-end GPU optimization
5. **How-It-Works.md** - Technical explanation of AdvTok

### Additional Pages (To Create)

The following pages are referenced in the Home page but need to be created:

- **Using-Demo-Script.md** - Guide for advtok_demo.py
- **Using-GUI-Application.md** - Guide for advtok_chat.py
- **Programmatic-Usage.md** - API reference and examples
- **Custom-Integration.md** - Integration guide
- **MDD-Explained.md** - Multi-valued Decision Diagrams explained
- **Chat-Templates.md** - Why chat templates are critical
- **State-Isolation.md** - Preventing contamination
- **Running-Tests.md** - Test suite documentation
- **Test-Coverage.md** - Coverage details
- **Contributing.md** - Contribution guidelines
- **Development-Setup.md** - Dev environment setup
- **Stability-Fixes.md** - Fixes documentation
- **Performance-Optimization.md** - Performance guide
- **API-Reference.md** - Complete API reference
- **Architecture.md** - System architecture
- **Original-Research.md** - ACL 2025 paper info
- **Responsible-Use.md** - Ethical guidelines
- **Citation-Guide.md** - How to cite
- **Security-Implications.md** - Security analysis
- **Troubleshooting.md** - Common issues and solutions

## Page Creation Order

Recommended order for creating pages:

1. **Home** (navigation hub)
2. **Installation-Guide** (setup instructions)
3. **Quick-Start** (getting started)
4. **RTX-5080-Setup** (GPU optimization)
5. **How-It-Works** (core concepts)
6. **Using-Demo-Script** (usage guide)
7. **Troubleshooting** (common issues)
8. **Original-Research** (citations)
9. **Responsible-Use** (ethics)
10. Other pages as needed

## Linking Between Pages

Wiki pages use this syntax for internal links:

```markdown
[Link Text](Page-Name)
```

Example:
```markdown
See the [Installation Guide](Installation-Guide) for setup instructions.
```

## Formatting Guidelines

All wiki pages follow this structure:

1. **Title** (H1 heading)
2. **Introduction** (brief description)
3. **Table of Contents** (for longer pages)
4. **Content Sections** (H2 and H3 headings)
5. **Code Examples** (with syntax highlighting)
6. **Next Steps** (links to related pages)
7. **Footer** (update date, links)

## Updating Wiki Pages

To update wiki pages after initial creation:

```bash
# Clone wiki
git clone https://github.com/watson0x90/advtok.wiki.git
cd advtok.wiki

# Edit pages
vim Home.md  # or your preferred editor

# Commit changes
git add Home.md
git commit -m "Update home page with new sections"
git push origin master
```

## Maintenance

### Regular Updates

- Update version numbers when releasing
- Add new troubleshooting entries as issues arise
- Keep performance benchmarks current
- Update links if repository structure changes

### Page Review Checklist

- [ ] All internal links work
- [ ] Code examples are tested
- [ ] Commands are up-to-date
- [ ] Screenshots (if any) are current
- [ ] No broken external links
- [ ] Formatting is consistent
- [ ] Grammar and spelling checked

## Sources

Wiki content is derived from:

- Repository README.md
- STABILITY_FIXES.md
- IMPROVEMENTS_SUMMARY.md
- CONTAMINATION_ANALYSIS.md
- GUI_CHAT_TEMPLATE_FIX.md
- advtok/tests/README.md
- Original AdvTok research paper

## License

Same as main repository (MIT License).

---

**Need help?** Open an issue: https://github.com/watson0x90/advtok/issues
