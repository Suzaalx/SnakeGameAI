# Assets Directory

This directory contains demo assets for the Snake Game AI project.

## Demo Files

To complete the project setup, add the following demo files:

### Required Demo GIFs

1. **training_demo.gif** - Shows the AI learning process during training
   - Recommended: 10-15 second clip showing initial random moves evolving to strategic play
   - Size: Optimized for web viewing (< 5MB)

2. **demo_performance.gif** - Shows trained agent achieving high scores
   - Recommended: 10-15 second clip of trained model playing effectively
   - Size: Optimized for web viewing (< 5MB)

### How to Create Demo GIFs

1. **Record Training Session**:
   ```bash
   # Start training and record screen
   python agent.py
   ```

2. **Record Demo Session**:
   ```bash
   # Run demo mode and record
   python demo.py --games 3 --speed 25
   ```

3. **Convert to GIF**:
   - Use screen recording software (QuickTime, OBS, etc.)
   - Convert to GIF using online tools or ffmpeg
   - Optimize file size for web

### Optional Assets

- **screenshots/** - Static screenshots of the game UI
- **models/** - Pre-trained model files for quick demo
- **charts/** - Performance charts and graphs

## File Naming Convention

- Use lowercase with underscores: `training_demo.gif`
- Include descriptive names: `high_score_gameplay.gif`
- Keep file sizes reasonable for GitHub (< 10MB per file)

## Usage in README

These assets are referenced in the main README.md file:

```markdown
![Training Demo](assets/training_demo.gif)
![Demo Performance](assets/demo_performance.gif)
```

Make sure the file paths match exactly for proper display.