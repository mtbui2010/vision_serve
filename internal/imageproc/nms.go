package imageproc

import (
	"sort"

	"visionserve/pkg/api"
)

// NMS (Non-Maximum Suppression) tiêu chuẩn theo IoU, dùng cho các model CẦN nó.
//
// LƯU Ý (CLAUDE.md): RF-DETR là NMS-free — KHÔNG gọi NMS cho RF-DETR. Hàm này
// là tiện ích dùng chung cho các model anchor/grid-based mà cộng đồng thêm sau.
func NMS(dets []api.Detection, iouThresh float64) []api.Detection {
	if len(dets) == 0 {
		return dets
	}
	idx := make([]int, len(dets))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return dets[idx[a]].Conf > dets[idx[b]].Conf })

	suppressed := make([]bool, len(dets))
	var keep []api.Detection
	for _, i := range idx {
		if suppressed[i] {
			continue
		}
		keep = append(keep, dets[i])
		for _, j := range idx {
			if j == i || suppressed[j] {
				continue
			}
			// chỉ suppress trong cùng class
			if dets[j].Class == dets[i].Class && iou(dets[i].BBox, dets[j].BBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// iou tính Intersection-over-Union giữa hai box dạng [x,y,w,h].
func iou(a, b [4]float64) float64 {
	ax1, ay1, ax2, ay2 := a[0], a[1], a[0]+a[2], a[1]+a[3]
	bx1, by1, bx2, by2 := b[0], b[1], b[0]+b[2], b[1]+b[3]

	ix1, iy1 := max(ax1, bx1), max(ay1, by1)
	ix2, iy2 := min(ax2, bx2), min(ay2, by2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a[2]*a[3] + b[2]*b[3] - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
