import cv2
import subprocess
import numpy as np
from collections import deque, defaultdict

# --- Settings ---
rows, cols = 5, 9
history_len = 30
frame_skip = 2
sub_h, sub_w = 10, 10
cell_size = 40
distance_threshold = 950

# Shape weighting
hu_weight = 5.0   # <<--- adjust this to change importance of Hu moments

# Shift subgrid inside each cell (used only for classification; display uses full cell)
shift_x = -6  # negative = left
shift_y = -5  # negative = up
sub_y = (cell_size - sub_h)//2 + shift_y
sub_x = (cell_size - sub_w)//2 + shift_x

# Phone resolution & grid coordinates in landscape
width, height = 2400, 1080
x1, y1 = 1010, 90
x2, y2 = 2330, 970
frame_size = width * height * 3

# Display settings
debug_scale = 3
arrow_thickness = 3
arrow_tip_len = 0.25
min_opacity = 0.2

# Swap persistence
swap_history_len = 30
swap_history = deque(maxlen=swap_history_len)

# --- Launch adb + ffmpeg pipeline ---
cmd = ["ffmpeg", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
adb_proc = subprocess.Popen(
    ["adb", "exec-out", "screenrecord", "--output-format=h264", "-"],
    stdout=subprocess.PIPE
)
ffmpeg_proc = subprocess.Popen(cmd, stdin=adb_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# --- Data structures ---
gem_vectors = []
gem_colors = []
history_vecs = [[deque(maxlen=history_len) for _ in range(cols)] for _ in range(rows)]

frame_count = 0
while True:
    raw_frame = ffmpeg_proc.stdout.read(frame_size)
    if not raw_frame:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    grid = frame[y1:y2, x1:x2]

    # Resize grid
    grid_small = cv2.resize(grid, (cols*cell_size, rows*cell_size), interpolation=cv2.INTER_AREA)
    cell_h = grid_small.shape[0] // rows
    cell_w = grid_small.shape[1] // cols

    debug_display_full = np.zeros((rows*cell_h, cols*cell_w, 3), dtype=np.uint8)
    gem_idx_grid = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols):
            y0, y1_cell = r*cell_h, (r+1)*cell_h
            x0, x1_cell = c*cell_w, (c+1)*cell_w

            cell = grid_small[y0:y1_cell, x0:x1_cell]
            if cell.size == 0:
                continue

            # classification uses subgrid
            subgrid = cell[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]

            # --- Color + Hu moments ---
            subgrid_f = subgrid.astype(np.float32).flatten()

            gray = cv2.cvtColor(subgrid, cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(gray)
            hu = cv2.HuMoments(moments).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            # Weight Hu contribution
            vec = np.concatenate([subgrid_f, hu_weight * hu])

            # temporal average
            history_vecs[r][c].append(vec)
            vec_avg = np.mean(history_vecs[r][c], axis=0)

            if len(gem_vectors) == 0:
                gem_vectors.append(vec_avg)
                gem_colors.append(tuple(np.random.randint(0,256,3).tolist()))
                gem_idx = 0
            else:
                distances = np.linalg.norm(np.array(gem_vectors) - vec_avg, axis=1)
                min_dist = distances.min()
                if min_dist < distance_threshold:
                    gem_idx = distances.argmin()
                else:
                    gem_vectors.append(vec_avg)
                    gem_colors.append(tuple(np.random.randint(0,256,3).tolist()))
                    gem_idx = len(gem_vectors)-1

            gem_idx_grid[r, c] = gem_idx
            debug_display_full[y0:y1_cell, x0:x1_cell] = cell

    # --- Detect valid swaps ---
    swaps = []
    for r in range(rows):
        for c in range(cols):
            if c < cols-1:
                gem_idx_grid[r, c], gem_idx_grid[r, c+1] = gem_idx_grid[r, c+1], gem_idx_grid[r, c]
                max_len = 0
                row_vals = gem_idx_grid[r, :]
                for i in range(cols-2):
                    if row_vals[i] == row_vals[i+1] == row_vals[i+2]:
                        length = 3
                        if i+3 < cols and row_vals[i] == row_vals[i+3]:
                            length = 4
                        max_len = max(max_len, length)
                for col in [c, c+1]:
                    col_vals = gem_idx_grid[:, col]
                    for i in range(rows-2):
                        if col_vals[i] == col_vals[i+1] == col_vals[i+2]:
                            length = 3
                            if i+3 < rows and col_vals[i] == col_vals[i+3]:
                                length = 4
                            max_len = max(max_len, length)
                if max_len >= 3:
                    swaps.append((r,c,r,c+1,max_len))
                gem_idx_grid[r, c], gem_idx_grid[r, c+1] = gem_idx_grid[r, c+1], gem_idx_grid[r, c]
            if r < rows-1:
                gem_idx_grid[r, c], gem_idx_grid[r+1, c] = gem_idx_grid[r+1, c], gem_idx_grid[r, c]
                max_len = 0
                col_vals = gem_idx_grid[:, c]
                for i in range(rows-2):
                    if col_vals[i] == col_vals[i+1] == col_vals[i+2]:
                        length = 3
                        if i+3 < rows and col_vals[i] == col_vals[i+3]:
                            length = 4
                        max_len = max(max_len, length)
                for row in [r, r+1]:
                    row_vals = gem_idx_grid[row, :]
                    for i in range(cols-2):
                        if row_vals[i] == row_vals[i+1] == row_vals[i+2]:
                            length = 3
                            if i+3 < cols and row_vals[i] == row_vals[i+3]:
                                length = 4
                            max_len = max(max_len, length)
                if max_len >= 3:
                    swaps.append((r,c,r+1,c,max_len))
                gem_idx_grid[r, c], gem_idx_grid[r+1, c] = gem_idx_grid[r+1, c], gem_idx_grid[r, c]

    swap_history.append(swaps)

    swap_counter = defaultdict(int)
    for hist_swaps in swap_history:
        for s in hist_swaps:
            swap_counter[s[:4]] += 1

        # --- Draw swaps with opacity ---
    overlay_full = debug_display_full.copy()
    arrow_layer = np.zeros_like(overlay_full, dtype=np.uint8)

    for (r1,c1,r2,c2,match_len) in swaps:
        freq = swap_counter[(r1,c1,r2,c2)] / len(swap_history)
        if freq < min_opacity:
            continue

        # Scale opacity between min_opacity..1
        alpha = freq  

        y1_overlay = r1*cell_h + cell_h//2
        x1_overlay = c1*cell_w + cell_w//2
        y2_overlay = r2*cell_h + cell_h//2
        x2_overlay = c2*cell_w + cell_w//2

        color = (0,0,255) if match_len == 3 else (255,0,0)  # red=3, blue=4+

        # draw on separate layer
        cv2.arrowedLine(
            arrow_layer, (x1_overlay, y1_overlay), (x2_overlay, y2_overlay),
            color, thickness=arrow_thickness, tipLength=arrow_tip_len
        )

        # blend only the bounding box around the arrow
        x_min, x_max = min(x1_overlay, x2_overlay), max(x1_overlay, x2_overlay)
        y_min, y_max = min(y1_overlay, y2_overlay), max(y1_overlay, y2_overlay)
        pad = 20
        x_min, x_max = max(0, x_min-pad), min(arrow_layer.shape[1], x_max+pad)
        y_min, y_max = max(0, y_min-pad), min(arrow_layer.shape[0], y_max+pad)

        roi_bg = overlay_full[y_min:y_max, x_min:x_max]
        roi_fg = arrow_layer[y_min:y_max, x_min:x_max]

        mask = cv2.cvtColor(roi_fg, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(bool)

        # blend only where arrow exists
        roi_bg[mask] = cv2.addWeighted(
            roi_bg[mask], 1-alpha, roi_fg[mask], alpha, 0
        )
        overlay_full[y_min:y_max, x_min:x_max] = roi_bg


    debug_full_scaled = cv2.resize(
        overlay_full, (overlay_full.shape[1]*debug_scale, overlay_full.shape[0]*debug_scale),
        interpolation=cv2.INTER_NEAREST
    )

    # --- Classification panel ---
    class_panel = debug_display_full.copy()
    for r in range(rows):
        for c in range(cols):
            y0, y1_cell = r*cell_h, (r+1)*cell_h
            x0, x1_cell = c*cell_w, (c+1)*cell_w
            overlay_color = np.full((cell_h, cell_w, 3), gem_colors[gem_idx_grid[r, c]], np.uint8)
            alpha = 0.4
            class_panel[y0:y1_cell, x0:x1_cell] = cv2.addWeighted(
                class_panel[y0:y1_cell, x0:x1_cell], 1-alpha,
                overlay_color, alpha, 0
            )
            cv2.putText(
                class_panel, str(gem_idx_grid[r, c]),
                (x0 + cell_w//3, y0 + int(cell_h*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
            )

    class_panel_scaled = cv2.resize(
        class_panel, (class_panel.shape[1]*debug_scale, class_panel.shape[0]*debug_scale),
        interpolation=cv2.INTER_NEAREST
    )

    # --- Show windows ---
    cv2.imshow("Debug FULL Cells with Swaps (BIG)", debug_full_scaled)
    cv2.imshow("Class Panel (plants + colors + IDs)", class_panel_scaled)

    if cv2.waitKey(1) & 0xFF == 27:
        break
