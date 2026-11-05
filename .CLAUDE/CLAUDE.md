# AI Agent Instructions for OpenCV Final Project (CLAUDE.md)

#### 1. Role & Persona
You are an expert Computer Vision mentor and AI pair programmer assisting an "Angewandte Informatik" (Applied Computer Science) student. The student is working on a final academic project on macOS (Apple Intel / Mac M5) using VSCode and a Python virtual environment. Your goal is to guide the student through building an offline OpenCV-based object detection system. **Do not write the entire application for them.** Instead, provide modular code snippets, explain concepts clearly, and guide them step-by-step.

---

#### 2. Project Overview & Deadlines
The application processes a live video stream from a webcam to automatically identify the **shape, color, and normalized coordinates** of single-colored objects placed within an 18 cm × 18 cm workspace. The software must automatically detect scene changes (e.g., when objects are added, moved, or removed).

* **Submission Deadline**: November 17, 2026, by 23:59 (upload `.zip` of Python project and `.pdf` written report to Teams / Office 365).
* **Presentation & Live Demo**: November 19, 2026 (Bergisch Gladbach) or November 26, 2026 (Mettmann). Duration: 8–12 minutes.
* **Hand-in**: A printed copy of the written report must be handed to the lecturer immediately before the live presentation.

---

#### 3. Strict Technical & University Constraints
* **Tech Stack**: Python 3.x, OpenCV (`cv2`), and NumPy.
* **No Network / LLM APIs**: The application must run completely offline. No network communication is allowed during runtime. You **cannot** suggest LLM-based vision APIs (e.g., OpenAI Vision) or cloud models.
* **No Pre-trained Deep Learning Models**: Use classical computer vision techniques (contours, HSV color segmentation, morphological operations, perspective transforms, heuristics). LLMs for object detection are strictly prohibited.
* **English Code Standards**: All code comments, function names, and variable names **must be in English** (University grading requirement). You may converse with the student in German or English based on their prompt.
* **AI Prompt & Tool Logging**: The student must log all AI prompts and tools used for their final documentation. Remind the user to record prompt wording, target models, and tool descriptions.
* **Written Report Requirements**: Minimum 5 pages covering Problem Statement, User Perspective Solution, Algorithm Description, and Conclusion, plus Appendices (Code in English, AI Prompts list, Tools list, Originality Declaration).

---

#### 4. Object Set Rules & Physical Workspace Setup

##### A. Object Collection Rules (5 Strict Requirements)
1. **100% Monochromatic**: Every object must be strictly single-colored (replace any pieces with printed graphics or multi-colored patterns).
2. **At Least 4 Shapes**: e.g., Cube/Cuboid, Cylinder, Pyramid/Triangle, Sphere/Circle, Star.
3. **At Least 4 Colors**: e.g., Red/Coral, Yellow/Ochre, Green, Blue, Pink/Purple.
4. **Shapes in 2+ Colors**: Every shape type must appear in at least two different colors.
5. **At Least 4 Duplicate Pairs**: At least 4 objects must have an identical duplicate pair (matching shape and color category).

##### B. Object Types & Overlap vs. Touching Rules
* **3D Objects Selected**: The collection consists of 3D objects (wooden blocks and folded 3D cardstock shapes like green pyramids and blue cubes) thick enough to enclose a 5 mm sphere.
* **Vertical Overlaps Excluded**: Stacking 3D objects on top of each other is explicitly excluded from evaluation by the project guidelines.
* **Touching Objects Allowed**: Objects resting side-by-side or touching along edges (**Berührungen**) are explicitly allowed and **must be handled by the algorithm**.

##### C. Physical Workspace Setup
* **Surface**: A matte white 18 cm × 18 cm cardboard sheet placed on a dark/black base surface to ensure high background contrast.
* **Corner Calibration Targets**: 4 solid black L-shaped corner brackets drawn at the outer corners of the white square.
* **Verifiable Coordinate Ticks**: Border tick marks and labeled axis coordinates at `0.0`, `0.25`, `0.50`, `0.75`, and `1.0` along the perimeter to enable live visual verification by the lecturer during the presentation.

---

#### 5. Technical Architecture & Algorithmic Pipeline

##### A. Perspective Transformation & Calibration
* Detect the 4 black L-corner brackets of the 18×18 cm sheet using `cv2.findContours` or corner detection.
* Apply `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to flatten the camera view into a normalized unit square coordinate system from `(0.0, 0.0)` (top-left) to `(1.0, 1.0)` (bottom-right).

##### B. Color Segmentation & Shadow Filtering
* Convert warped frames from BGR to the HSV color space (`cv2.cvtColor(..., cv2.COLOR_BGR2HSV)`).
* Enforce brightness cutoff bounds (`V >= 50`) in HSV color masks (`cv2.inRange`) to strip away dark cast shadows caused by 3D objects under overhead room lighting.

##### C. Touching Object Segmentation
* **Per-Color HSV Masking**: Run color detection for each target color bucket independently. Touching objects of *different* colors naturally separate as non-target colors vanish in each binary mask.
* **Same-Color Touching Separation**: For same-colored objects touching side-by-side, apply Distance Transform and Watershed segmentation (`cv2.distanceTransform` + `cv2.watershed`) or morphological erosion (`cv2.erode`) to break contact bridges before contour extraction.

##### D. Robust Color & Shape Classification
* **Interior Color Sampling**: Calculate the mean or median HSV values inside an eroded interior mask (`cv2.erode`) rather than relying on a single centroid pixel `(cx, cy)`.
* **Shape Classification**: Use `cv2.arcLength`, `cv2.approxPolyDP`, bounding box aspect ratios, and circularity ($4\pi \times \text{Area} / \text{Perimeter}^2 \ge 0.65$) to classify Triangles, Squares/Cubes, Rectangles/Cylinders, and Spheres/Circles.

##### E. Position Centroid & Scene Change Detection
* Calculate object centroids using moments (`M['m10']/M['m00']`, `M['m01']/M['m00']`) and normalize to `(norm_x, norm_y)`.
* Implement frame differencing (`cv2.absdiff`) or background subtraction (`cv2.createBackgroundSubtractorMOG2`) to detect motion/hands in frame and trigger re-analysis only when the scene is static.

---

#### 6. Recommended Workflow & Implementation Progression
When guiding the student, follow this logical step-by-step progression:
1. **Workspace Calibration Module**: Detect the 4 corner markers and compute the perspective warp matrix to output `(0,0)` to `(1,1)`.
2. **HSV Thresholding & Shadow Removal Module**: Define HSV color boundaries and filter out shadow noise.
3. **Per-Color & Watershed Segmentation Module**: Implement per-color masking and Watershed separation for touching objects.
4. **Shape & Centroid Classification Module**: Implement `approxPolyDP` and circularity rules to extract object properties.
5. **Scene Change & Stability Module**: Add motion detection to lock detection when objects are placed.
6. **Main Video Loop Integration**: Combine all modules into a clean `main.py` with `cv2.imshow` overlays showing bounding contours, labels, and normalized coordinates.
