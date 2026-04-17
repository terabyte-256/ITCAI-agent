# Design System Specification: The Academic Curator

## 1. Overview & Creative North Star
**Creative North Star: "The Digital Curator"**
This design system moves away from the "utility-first" look of standard SaaS platforms and moves toward a high-end, editorial experience. It is designed to feel like a prestigious university’s digital archive—authoritative, intelligent, and calm. 

Instead of rigid grids and heavy borders, we utilize **Intentional Asymmetry** and **Tonal Depth**. The UI should feel like layers of fine parchment and frosted glass stacked atop a deep, scholarly foundation. We break the "template" look by using exaggerated typographic scales and overlapping elements that suggest a human, curated touch rather than a machine-generated list.

---

## 2. Colors & Surface Architecture
The palette is rooted in academic tradition (Navy and Gold) but executed with modern digital depth.

### The "No-Line" Rule
**Borders are a failure of hierarchy.** In this system, 1px solid strokes for sectioning are prohibited. Boundaries must be defined through:
- **Background Shifting:** A `surface-container-low` section sitting on a `surface` background.
- **Tonal Transitions:** Using subtle shifts in the gray scale to denote a change in context.

### Surface Hierarchy & Nesting
Treat the interface as a physical desk.
- **Base Layer:** `surface` (#f9f9f9) is your primary canvas.
- **Nested Containers:** Use `surface-container-low` for large content areas and `surface-container-lowest` (#ffffff) for active cards to create a "lifted" effect.
- **The "Glass & Gradient" Rule:** Floating elements (like navigation or active chat bubbles) should use Glassmorphism. Apply `surface` with 80% opacity and a `24px` backdrop-blur to allow the university’s signature navy or gold to bleed through subtly.

### Signature Textures
Main CTAs and high-level headers should use a **"Scholarly Gradient"**: A linear transition from `primary` (#000a1e) to `primary-container` (#002147). This adds a "soul" to the digital interface that flat navy cannot provide.

---

## 3. Typography: The Editorial Voice
We pair the modern precision of **Inter** with the intellectual sophistication of **Noto Serif**.

*   **Display & Headlines (Noto Serif):** Use `display-lg` and `headline-lg` to create an editorial feel. These should be set with tighter letter-spacing (-0.02em) to feel like a premium journal.
*   **Body (Inter):** All functional text uses Inter. It provides a neutral, highly readable counterpoint to the serif headings.
*   **Hierarchy as Brand:** Use `title-lg` (Inter, Medium weight) for card titles to ensure clarity, but always introduce a page with a `display-md` (Noto Serif) to establish authority.

---

## 4. Elevation & Depth
We convey importance through **Tonal Layering**, not structural lines.

*   **The Layering Principle:** To make a card "pop," do not add a border. Place a `surface-container-lowest` card on top of a `surface-container-high` background. The natural contrast creates the edge.
*   **Ambient Shadows:** For floating modals or the "Campus Agent" chat interface, use an extra-diffused shadow: `box-shadow: 0 20px 50px rgba(0, 27, 61, 0.05)`. Note the use of a Navy tint (`on-primary-fixed`) in the shadow rather than pure black.
*   **The "Ghost Border":** If accessibility requires a container edge, use `outline-variant` at 15% opacity. It should be felt, not seen.
*   **Glassmorphism:** Use for the Sidebar. It should feel like a pane of glass over the campus map or main content, utilizing `backdrop-filter: blur(12px)`.

---

## 5. Components

### Navigation Sidebar
- **Styling:** Use `surface-container-low` with a subtle `primary` gradient bleed at the bottom.
- **Interaction:** Active states should not use a box; use a "Gold Thread"—a 3px vertical line of `secondary` (#735c00) on the far left of the item.

### The Knowledge Card
- **Forbid Dividers:** Use `spacing-xl` to separate headers from body text within cards.
- **Background:** `surface-container-lowest`.
- **Corner Radius:** Use `xl` (0.75rem) to soften the academic "hardness."

### Chat Bubbles (The Agent Interface)
- **Agent Bubble:** `primary-container` (#002147) with `on-primary` text. Use an asymmetrical radius: `top-left: none`, others `lg`.
- **User Bubble:** `surface-container-highest` with `on-surface`. This keeps the focus on the Agent’s "Knowledge."

### Buttons
- **Primary:** Scholarly Gradient (Navy) with `on-primary` text. No border. `md` roundedness.
- **Secondary (The "Gold Accent"):** `secondary-container` background with `on-secondary-container` text. Use for high-importance academic actions (e.g., "Apply," "Enroll").

### Input Fields
- **Styling:** A simple `surface-container-highest` fill. No bottom line. No border. On focus, transition the background to `surface-container-lowest` and add a `primary` ghost-border.

---

## 6. Do’s and Don’ts

### Do:
- **Do** use large amounts of white space to suggest "room for thought."
- **Do** use `notoSerif` for rhetorical questions or quotes from the Knowledge Agent.
- **Do** use asymmetric layouts (e.g., a sidebar that is 25% width and a main card that is 60% width, off-center) to feel modern and high-end.

### Don't:
- **Don't** use 1px solid dividers to separate list items. Use vertical padding and background shifts.
- **Don't** use pure black (#000000). Always use `primary` (#000a1e) or `on-surface` (#1a1c1c) for text.
- **Don't** over-use the Gold. Gold is a "surgical" accent; use it only for CTAs, notifications, or critical milestones.