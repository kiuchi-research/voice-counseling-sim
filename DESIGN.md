---
version: alpha
name: Template Vibe Coding House Style
description: Shared visual identity for UI, dashboards, charts, tables, diagrams, and paper-ready figures produced from this AI development template.
colors:
  primary: "#0B0F14"
  secondary: "#475569"
  tertiary: "#0F766E"
  neutral: "#F8FAFC"
  ink-950: "#0B0F14"
  ink-900: "#111827"
  ink-800: "#1F2937"
  ink-700: "#334155"
  ink-600: "#475569"
  ink-500: "#64748B"
  line-300: "#CBD5E1"
  line-200: "#E2E8F0"
  surface-0: "#FFFFFF"
  surface-50: "#F8FAFC"
  surface-100: "#F1F5F9"
  surface-200: "#E8EEF5"
  surface-900: "#0F172A"
  teal-700: "#0F766E"
  teal-600: "#0D9488"
  teal-500: "#14B8A6"
  cobalt-700: "#1D4ED8"
  cobalt-600: "#2563EB"
  cobalt-500: "#60A5FA"
  rust-700: "#B45309"
  rust-600: "#D97706"
  rust-500: "#F59E0B"
  success: "#3F6212"
  warning: "#B45309"
  danger: "#BE123C"
typography:
  display:
    fontFamily: IBM Plex Sans
    fontSize: 48px
    fontWeight: 650
    lineHeight: 1.17
    letterSpacing: 0em
  h1:
    fontFamily: IBM Plex Sans
    fontSize: 36px
    fontWeight: 650
    lineHeight: 1.22
    letterSpacing: 0em
  h2:
    fontFamily: IBM Plex Sans
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.29
    letterSpacing: 0em
  h3:
    fontFamily: IBM Plex Sans
    fontSize: 22px
    fontWeight: 600
    lineHeight: 1.36
    letterSpacing: 0em
  body-lg:
    fontFamily: IBM Plex Sans
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.625
    letterSpacing: 0em
  body:
    fontFamily: IBM Plex Sans
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0em
  small:
    fontFamily: IBM Plex Sans
    fontSize: 13px
    fontWeight: 450
    lineHeight: 1.54
    letterSpacing: 0em
  caption:
    fontFamily: IBM Plex Sans
    fontSize: 12px
    fontWeight: 450
    lineHeight: 1.5
    letterSpacing: 0em
  mono:
    fontFamily: IBM Plex Mono
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.54
    letterSpacing: 0em
  editorial-heading:
    fontFamily: Source Serif 4
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: 0em
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  3xl: 64px
  4xl: 96px
rounded:
  sm: 6px
  md: 10px
  lg: 16px
  pill: 999px
components:
  button-primary:
    backgroundColor: "{colors.teal-700}"
    textColor: "{colors.surface-0}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: 12px 16px
  button-secondary:
    backgroundColor: "{colors.surface-0}"
    textColor: "{colors.ink-800}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: 12px 16px
  panel:
    backgroundColor: "{colors.surface-0}"
    textColor: "{colors.ink-900}"
    rounded: "{rounded.md}"
    padding: 24px
  dashboard-panel:
    backgroundColor: "{colors.surface-50}"
    textColor: "{colors.ink-900}"
    rounded: "{rounded.sm}"
    padding: 16px
  code-block:
    backgroundColor: "{colors.surface-100}"
    textColor: "{colors.ink-900}"
    typography: "{typography.mono}"
    rounded: "{rounded.sm}"
    padding: 16px
---

# DESIGN.md

This file is the default design contract for this repository and for downstream projects created from it.
Use it as a house style for UI, dashboards, charts, tables, and paper-ready figures.

## 1. Design Intent

Build interfaces and figures that feel:

- technical, calm, and exact
- editorial rather than flashy
- modern but not trend-chasing
- dense enough for serious work, never crowded
- distinctive through typography, spacing, and composition instead of gimmicks

The default visual language is a restrained neutral foundation with one accent family.
Do not imitate a specific commercial brand. Build a reusable house style.

## 2. Priority Rules

When generating design or code, follow this order:

1. User task requirements
2. Artifact type selection
3. Mode selection
4. Shared tokens and component rules
5. Chart, table, or diagram rules when the output is data-centric

If rules conflict:

- publisher, journal, or venue submission requirements override this file for manuscript-bound assets
- chart rules override generic UI styling for charts
- table rules override generic card styling for tables
- paper-editorial rules override dashboard styling for manuscript figures

## 3. Artifact Types

Choose the closest artifact type first.

### Product UI

Use for:

- app pages
- settings
- forms
- documentation screens
- CRUD interfaces

Default mode: `product-ui`

### Dashboard

Use for:

- analytics
- monitoring
- model evaluation views
- experiment comparison
- operational metrics

Default mode: `dashboard`

### Paper / Report

Use for:

- manuscript figures
- technical reports
- appendices
- methods diagrams
- result tables

Default mode: `paper-editorial`

### Figure vs Table vs Text

Decide the artifact before styling it.

- Use a figure for trends, structure, flow, comparisons, distributions, timelines, and conceptual relationships.
- Use a table for exact values, coefficients, confidence intervals, sample sizes, variable lists, and method detail.
- Use prose when the message can be stated cleanly in one or two sentences.
- For manuscripts, keep only the argument-carrying minimum in the main text and move extra robustness, long tables, or secondary detail to supplement.

### Showcase / Landing

Use for:

- project front pages
- hero sections
- feature overviews
- announcement pages

Default mode: `showcase`

Do not use `showcase` styling for dense analytical content.

## 4. Shared Principles

### 4.1 Composition

- Prefer strong hierarchy over decoration.
- Use fewer, larger layout decisions instead of many small effects.
- Let whitespace create rhythm.
- Use asymmetry carefully: offset blocks, staggered sections, or uneven columns are allowed when the page remains readable.
- Prefer clear sectional framing with spacing and borders over filled background boxes everywhere.

### 4.2 Color Strategy

- Use a mostly neutral base.
- Use one accent family per artifact.
- Reserve saturated color for actions, highlights, annotations, and key series.
- Avoid rainbow palettes in UI.
- In charts, color may encode categories, but must still remain disciplined.

### 4.3 Surfaces

- Most surfaces should be flat or near-flat.
- Prefer border plus subtle shadow over heavy elevation.
- Avoid glassmorphism, blurred chrome, and over-layered translucency unless explicitly requested.

### 4.4 Typography

- Typography should carry much of the design character.
- Use sans for UI, mono for code or numeric detail, serif selectively for editorial artifacts.
- Prefer sentence case, not all caps.
- Tighten large headlines slightly; keep body text comfortable and neutral.

### 4.5 Data Display

- Data should read before decoration.
- Favor direct labeling, annotations, and explanatory captions.
- Use consistent numeric formatting within the same artifact.
- Reduce clutter before reducing information value.

## 5. Shared Tokens

### 5.1 Neutral Palette

- `ink-950`: `#0B0F14`
- `ink-900`: `#111827`
- `ink-800`: `#1F2937`
- `ink-700`: `#334155`
- `ink-600`: `#475569`
- `ink-500`: `#64748B`
- `line-300`: `#CBD5E1`
- `line-200`: `#E2E8F0`
- `surface-0`: `#FFFFFF`
- `surface-50`: `#F8FAFC`
- `surface-100`: `#F1F5F9`
- `surface-200`: `#E8EEF5`
- `surface-900`: `#0F172A`

### 5.2 Accent Families

Pick one family per artifact unless the user asks otherwise.

#### Teal family

- `accent-700`: `#0F766E`
- `accent-600`: `#0D9488`
- `accent-500`: `#14B8A6`
- Best for: default product UI, scientific calm, general-purpose interfaces

#### Cobalt family

- `accent-700`: `#1D4ED8`
- `accent-600`: `#2563EB`
- `accent-500`: `#60A5FA`
- Best for: dashboards, enterprise tools, infrastructure interfaces

#### Rust family

- `accent-700`: `#B45309`
- `accent-600`: `#D97706`
- `accent-500`: `#F59E0B`
- Best for: editorial pages, report highlights, warm but serious presentation

Never mix teal and cobalt in the same page chrome.
Warm accent may appear in figures or callouts, but do not let it dominate action UI unless the page is explicitly editorial.

### 5.3 Semantic Colors

- `success`: `#3F6212`
- `warning`: `#B45309`
- `danger`: `#BE123C`
- `info`: use the active accent family

### 5.4 Typography Stack

#### UI Sans

- Primary: `IBM Plex Sans`
- Secondary: `Geist`
- Fallback: `ui-sans-serif, sans-serif`

#### Editorial Serif

- Primary: `Source Serif 4`
- Fallback: `Georgia, serif`

#### Mono

- Primary: `IBM Plex Mono`
- Fallback: `ui-monospace, SFMono-Regular, monospace`

### 5.5 Type Scale

Use these as intent levels, not rigid requirements.

- Display: `48/56`, weight `600-700`
- H1: `36/44`, weight `600-700`
- H2: `28/36`, weight `600`
- H3: `22/30`, weight `600`
- H4: `18/26`, weight `600`
- Body large: `16/26`, weight `400-500`
- Body: `15/24`, weight `400`
- Small: `13/20`, weight `400-500`
- Caption: `12/18`, weight `400-500`
- Code / numeric detail: `13/20`, mono

For paper figures and tables, use smaller but still legible text:

- Axis / labels: `8-9 pt`
- Tick labels: `7-8 pt`
- Table body: `8-9 pt`
- Caption: `8-9 pt`

### 5.6 Spacing Scale

Use a 4 px base unit.

- `4, 8, 12, 16, 24, 32, 48, 64, 96`

Default layout rhythm:

- inner component padding: `12-24`
- card padding: `16-24`
- section gap: `48-96`
- dense dashboard gap: `16-24`

### 5.7 Radius

- small: `6`
- medium: `10`
- large: `16`
- pill: `999`

Avoid mixing many radii in one artifact.

### 5.8 Borders and Shadows

- Standard border: `1 px solid line-200 or line-300`
- Emphasis border: `1.5-2 px`
- Standard shadow: very soft, short, low blur
- Do not stack more than one visible shadow layer on common cards

## 6. Mode Profiles

### 6.1 `product-ui`

Use when the output is a general application interface.

Atmosphere:

- clean
- precise
- lightly editorial
- calm confidence

Structure:

- mostly light surfaces
- generous whitespace
- one strong page header, then modular sections
- forms and tables should feel purposeful, not consumer-app cute

Preferred accent families:

- teal by default
- cobalt when the product is more enterprise or infrastructure oriented

Typography:

- sans throughout
- mono for code snippets, IDs, metrics, and timestamps

Components:

- buttons are compact, solid, and quiet
- cards have borders and minimal shadow
- tabs are understated and line-based
- empty states rely on layout and copy, not illustration-heavy filler

Avoid:

- oversized rounded blobs
- gradient overload
- loud glass effects
- generic AI-app purple aesthetic

### 6.2 `dashboard`

Use for information-dense analytical screens.

Atmosphere:

- rigorous
- compact
- data-first
- operational

Structure:

- tighter spacing than product UI
- clear panel framing
- strong use of tabular numbers
- sections grouped by task or metric family

Surface choices:

- light dashboard is preferred by default
- dark dashboards are allowed only when the task explicitly benefits from a control-room feel

Preferred accent family:

- cobalt by default
- teal is acceptable for research dashboards

Typography:

- sans plus mono
- avoid serif in the dashboard chrome

Components:

- stat cards should privilege the main metric, delta, and context
- filters should be compact and aligned
- charts should share scales and visual grammar where comparisons matter

Avoid:

- decorative hero sections inside dashboards
- too many competing panel backgrounds
- gauges unless the user explicitly asks for them

### 6.3 `paper-editorial`

Use for manuscripts, reports, and publication-ready outputs.

Atmosphere:

- print-aware
- minimal
- disciplined
- academic but not old-fashioned

Structure:

- white background by default
- serif may appear in headings or section titles
- figures and tables should feel consistent across the full document
- emphasis comes from alignment, spacing, and annotation

Preferred accent family:

- teal or rust
- use cobalt sparingly in paper figures unless there is a strong reason

Typography:

- serif for document headings is allowed
- sans for charts, tables, captions, and labels
- mono only for code fragments or parameter names

Submission-safe defaults when venue is unknown:

- plan around single-column widths near `85-90 mm` and double-column widths near `170-180 mm`
- aim for `7-9 pt` figure text at final output size
- prefer editable text and vector masters for charts, diagrams, and composite figure layouts
- keep original raster files for photographs, microscopy, and screenshots
- design legends and notes so they can move outside the figure when a venue requires separate captions
- if font compatibility is uncertain, fall back to `Arial` or `Helvetica` for final exported manuscript figures

Production hygiene:

- retain source data
- retain plotting code or an exact plotting procedure
- retain editable layout masters
- retain final exported submission assets
- retain revision history for figure and table changes

Avoid:

- dark mode figures for papers
- thick card shadows
- decorative gradients
- ornamental icons

### 6.4 `showcase`

Use for landing pages, project intros, and high-level overviews.

Atmosphere:

- bold but controlled
- high-contrast
- intentional
- memorable without becoming theatrical

Structure:

- dramatic typography is allowed in hero regions
- combine large headings, concise copy, and one or two graphic moves
- subsequent content sections should relax back toward `product-ui`

Preferred accent family:

- teal or rust

Allowed treatments:

- gradient fields with very low complexity
- large geometric backdrops
- split layouts
- staggered content entry

Avoid:

- turning the full site into a marketing page
- long pages where every section tries to be a hero
- decorative motion that obscures content

## 7. Component Rules

### 7.1 Buttons

- Primary button: solid accent fill, high contrast text, modest radius
- Secondary button: neutral surface, visible border, no heavy fill
- Tertiary button: text or ghost style
- Keep button labels short and verb-led

### 7.2 Cards and Panels

- Use borders first, shadows second
- Card headers should contain one clear title and optional supporting metadata
- Dense panels should not use oversized padding
- Large analytical panels may use subtle tinted headers or top rules

### 7.3 Forms

- Inputs should feel exact and stable
- Use visible labels outside the field, not placeholder-only labeling
- Error states should rely on text plus color
- Keep forms left-aligned and rhythmically spaced

### 7.4 Navigation

- Keep navigation visually quiet
- Use hierarchy through weight and spacing, not many colors
- Highlight the active item with a line, shape, or local tint

### 7.5 Code and Numeric Blocks

- Use mono
- Prefer flat surfaces with a thin border
- Long IDs, metrics, and timestamps should align cleanly
- Use tabular numerals where supported

## 8. Chart Rules

Apply these rules to all data visualizations unless the user explicitly requests a different visual grammar.

### 8.1 General Intent

- Charts should read clearly in both screen and export contexts.
- Favor explanation over spectacle.
- Design for comparison, not ornament.
- One figure should carry one primary message.
- If the main need is exact lookup rather than pattern recognition, use a table instead.
- If the point can be said cleanly in one or two sentences, prose may be better than a chart.
- Main-text figures should carry the argument; overflow detail belongs in supplement.

### Panel Logic

- Arrange multipanel figures in natural reading order and label them consistently, for example `a`, `b`, `c`.
- Align comparable panels in scale, axis placement, and label logic.
- Give the most important panel more space.
- Reduce empty space, but do not crowd labels or panels.
- A figure should remain readable when reduced to roughly half size.

### 8.2 Chart Background and Frame

- Default background: white
- UI dashboards may use `surface-50`
- Paper figures should stay on white
- Remove heavy outer borders unless needed for embedding

### 8.3 Preferred Chart Types

Prefer:

- line charts
- bar charts
- dot plots
- scatter plots
- box plots
- violin plots when distribution shape matters
- heatmaps
- small multiples

Use cautiously:

- stacked areas
- stacked bars
- pies or donuts only for 4 categories or fewer and only when part-to-whole is the main message

Avoid unless explicitly requested:

- 3D charts
- radar charts
- gauges
- dual-axis charts
- chartjunk and decorative illustration inside plots

### 8.4 Series Palette

For charts, use this disciplined categorical palette:

1. `#0F766E`
2. `#1D4ED8`
3. `#B45309`
4. `#BE123C`
5. `#475569`
6. `#3F6212`

Rules:

- up to 3 series: use the first 3 colors
- more than 5 series: introduce dash patterns, marker shapes, or small multiples
- one highlighted series may use the active accent; comparison series should recede

### 8.5 Axes, Grid, and Labels

- Use sentence case
- Include units in axis labels or subtitle, not redundantly everywhere
- State sample size when it matters to interpretation
- Grid lines should be thin and low contrast
- Use grid lines primarily on the quantitative axis
- Start bar charts at zero unless there is a very strong documented reason not to
- Ticks should be sparse but informative
- Do not crop scales to exaggerate the claim
- Remove nonessential background grid lines

Recommended colors:

- axes: `ink-700`
- tick labels: `ink-600`
- grid lines: `line-200`

### 8.6 Lines, Marks, and Bars

- Standard line width: `1.75-2.5 px` on screen, slightly thinner in print
- Comparison series may be thinner or lighter
- Markers should appear only when they help track points or categories
- Bars should have minimal or no border
- Use rounded bar corners sparingly; square bars are preferred in papers
- Prefer line type, marker shape, position, and weight over colored text for emphasis

### 8.7 Legends and Direct Labels

- Prefer direct labels when there are few series
- Use legends outside the plotting area when direct labeling harms readability
- Put legend at top or right, not floating over dense data

### 8.8 Annotation

- Annotate peaks, changes, outliers, thresholds, or policy-relevant results
- Use short annotation copy
- Annotation color should be neutral or accent, never random
- Reference lines should be thin and clearly labeled

### 8.9 Uncertainty and Statistics

- Use confidence intervals, error bars, or ribbons when uncertainty matters
- Distinguish estimate from interval through line weight and opacity
- Statistical significance markers should be minimal and explained in caption or note
- Always define whether error bars or intervals represent `SD`, `SE`, `CI`, or another quantity

### 8.10 Heatmaps

- Use perceptually orderly scales
- Sequential data: light to dark within one hue family
- Diverging data: two balanced hues with a meaningful midpoint
- Never use neon rainbow heatmaps

### 8.11 Figure Sizes for Papers

Target print-friendly sizes:

- single column: about `85 mm` wide
- wide single column / mid layout: about `120-130 mm` wide
- double column: about `178 mm` wide

Typical figure height:

- `55-120 mm`

Design figures so labels remain legible when reduced to final print size.

### 8.12 Accessibility and Grayscale

- Never rely on color alone
- Use shape, dash, pattern, label, or ordering as a backup encoding
- Check that the figure still works in grayscale

### 8.13 Scientific Images and Screenshots

- Prefer scale bars over magnification numbers when showing scientific images.
- Cropping is acceptable when it preserves meaning and context.
- Brightness and contrast adjustments must be consistent across the whole image, not selectively retouched.
- Mark composites, splices, or stitched panels clearly.
- Keep original source images separately from presentation-ready exports.
- Do not use generative fill, content-aware reconstruction, or synthetic image completion unless venue policy explicitly allows it and disclosure is handled.

## 9. Table Rules

Apply these rules to UI data tables and paper tables, then adjust density by mode.

### 9.1 General Intent

- Tables exist for exact lookup, comparison, and auditability
- If the point is trend or pattern, prefer a chart
- If the point is exact values or method detail, prefer a table
- If the table is tiny and obvious, prefer prose instead

### 9.2 Structure

- Use clear column groups when the schema is complex
- Keep the first column stable and descriptive
- Put units in headers, not repeated in every cell
- Use footnotes or notes for methodological caveats
- Put columns intended for comparison adjacent to one another

### 9.3 Alignment

- text: left align
- counts and currency: right align
- decimals: align by decimal point when feasible
- short status tokens: centered only when this improves scanability

### 9.4 Rules and Fill

- Avoid vertical rules in paper-style tables
- Use one strong top rule, one header divider, and one bottom rule
- Dense UI tables may use subtle row dividers
- Zebra striping is optional in UI tables and should be very light

### 9.5 Typography

- Use sans for most tables
- Use tabular numerals when possible
- Header row should be slightly heavier, not dramatically larger
- Keep table text smaller than surrounding prose but never tiny

### 9.6 Emphasis

- Emphasize with weight, tint, or border, not many colors
- Highlight at most one primary column or one key row group
- Negative values and warnings may use semantic color carefully

### 9.7 Missing Values and Precision

- Use a consistent missing-value token such as `N/A` or `-`
- Do not mix precision within the same column unless necessary
- Default numeric precision should reflect the domain, not arbitrary long decimals

### 9.8 Paper Tables

- Caption goes above the table
- Notes go below the table
- Keep journal-like discipline: minimal lines, strong alignment, no decorative fills
- If the table is too wide, restructure before shrinking the font aggressively
- Notes should explain abbreviations, statistical symbols, missing-value conventions, and model adjustments in a consistent order across the manuscript

### 9.9 UI Tables

- Sticky headers are encouraged for long tables
- Row hover states should be subtle
- Sorting and filtering controls should be visible but quiet
- Selected rows should use tint plus outline, not only tint

## 10. Diagram Rules

Apply to flowcharts, pipelines, method diagrams, architecture diagrams, and concept maps.

### 10.1 Overall Style

- clean nodes
- limited shapes
- restrained color
- explicit relationships

### 10.2 Node Styling

- Primary node shape: rounded rectangle
- Optional alternate shapes: pill for status, rectangle for process, circle only for compact state markers
- Default node fill: white or very light neutral
- Default node border: `1-1.5 px`

### 10.3 Connectors

- Use straight or gently orthogonal connectors
- Arrowheads should be simple and small
- Avoid tangled curves when layout can solve the problem
- Label arrows only when the relationship is not obvious

### 10.4 Grouping

- Group related nodes with whitespace first
- Use tinted background containers sparingly
- Each group should have a clear label

### 10.5 Diagram Density

- Keep method diagrams compact but not cramped
- Prefer multiple simple panels over one overstuffed mega-diagram
- If the diagram is part of a paper, ensure it survives grayscale printing

## 11. Motion Rules

Only relevant for web UI and interactive artifacts.

- Motion should clarify structure, not entertain by default
- Favor fade, slide, reveal, and stagger over bounce or spring-heavy novelty
- Keep durations short and consistent
- Disable or soften motion for dense analytical contexts

Recommended rhythm:

- micro interactions: `120-180 ms`
- panel and section transitions: `180-280 ms`

Avoid:

- looping decorative animation
- parallax in analytical products
- motion that competes with reading

## 12. Accessibility and Output Quality

- Body text contrast should meet a strong accessibility threshold
- Interactive targets should be comfortably clickable on mobile
- Focus states must be visible and intentional
- Do not encode meaning only by color
- Long tables and charts should remain readable when exported or printed

For paper and report outputs:

- assume white background export
- avoid thin pale lines that vanish in print
- verify grayscale readability

### 12.1 Submission-Safe Export Checks

- preserve vector output for line art, charts, and diagrams whenever possible
- verify raster resolution for photo-based panels before submission export
- confirm that text remains editable or that font substitution will not break the figure
- separate captions from figure art when the venue requires that workflow
- use predictable file naming such as `Fig1`, `Fig2`, `Table1` when preparing submission assets

## 13. Anti-Patterns

Do not generate:

- default purple-on-white SaaS styling
- random gradient backgrounds with no structural role
- heavy glassmorphism
- dashboards made of many identical floating cards
- paper figures with dark backgrounds
- tables with thick full grids unless the user explicitly requests spreadsheet aesthetics
- charts with too many colors and no semantic logic
- decorative icons or illustrations that crowd serious analytical content

## 14. Agent Instructions

When the user asks for "make it look better" without specifying a mode:

1. choose `product-ui` for normal application screens
2. choose `dashboard` for metric-heavy or evaluation-heavy screens
3. choose `paper-editorial` for manuscript, report, figure, or table outputs
4. choose `showcase` only for landing pages or hero sections

When the task mixes UI and data:

- use the selected UI mode for layout and chrome
- use chart or table rules for the data region

When the task is ambiguous:

- default to light theme
- use teal accent family
- choose cleaner and quieter over louder and trendier

When the task is manuscript-bound and the venue is unspecified:

- use `paper-editorial`
- prefer journal-safe sizing and typography
- avoid effects that fail in print or grayscale
- preserve editable source assets whenever possible

## 15. Prompt Guide

Use prompts like:

- "Use `product-ui` mode with the shared neutral palette and teal accent family."
- "Use `dashboard` mode. Keep the chrome restrained and let charts carry emphasis."
- "Use `paper-editorial` mode. Make the figure print-safe, white background, and grayscale-safe."
- "Use the table rules from `DESIGN.md`, with journal-style minimal lines and aligned decimals."
- "Use `showcase` only for the hero block, then relax the rest of the page into `product-ui`."

If multiple outputs are produced in one task, keep them in the same family so they look related across UI, charts, and tables.
