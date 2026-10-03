// Package rfdetr is a forwarding shim kept only for its import path: RF-DETR (and
// RT-DETR) now live in internal/models/detr, which registers the "rf-detr" and "rt-detr"
// architectures. Blank-importing this package still registers them (textalign's tests
// do); new code should import internal/models/detr or go through models.New("rf-detr").
package rfdetr

import _ "visionserve/internal/models/detr"
