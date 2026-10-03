package server

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"visionserve/internal/models"
	"visionserve/pkg/api"
)

// fullRequest sets EVERY option of api.PredictJSONRequest to a non-default value, so the
// content-type comparison below covers each one. Adding a field to api.PredictJSONRequest fails
// TestEveryOptionMapsOneToOne until it is given a value here.
func fullRequest(t *testing.T) (api.PredictJSONRequest, []byte, []byte) {
	img := pngBytes(t, 16, 12)
	depth := make([]uint16, 4*3)
	for i := range depth {
		depth[i] = uint16(1000 * (i + 1)) // no zeros: 0 means "no measurement" (NaN)
	}
	depthRaw := u16bytes(depth...)
	return api.PredictJSONRequest{
		Model:          "m",
		ImageBase64:    b64(img),
		Prompt:         "cat. remote.",
		Box:            "10,8,4,2;3,3,2,2",
		Point:          "5,6;7,8,0",
		MinSize:        0.5,
		MaxSize:        90,
		GripperMin:     3,
		GripperMax:     40.5,
		BoxThreshold:   0.35,
		TextThreshold:  0.2,
		BgMaxArea:      50,
		FgMinArea:      0.1,
		GridSize:       1000, // clamped to maxGridSize
		Method:         "sam",
		ClaimThreshold: 0.6,
		CropTemp:       0.05,
		ROI:            "2,2,12,10",
		Dilate:         -2,
		DepthBase64:    b64(depthRaw),
		DepthDtype:     "uint16",
		DepthWidth:     4,
		DepthHeight:    3,
		TemplateName:   "tpl",
		Encoding:       "base64",
	}, img, depthRaw
}

// formOf renders a request as multipart field values by json name (the reflection mirror of
// json.Marshal), flagging any field the sample leaves at its zero value.
func formOf(t *testing.T, q api.PredictJSONRequest) map[string]string {
	t.Helper()
	out := map[string]string{}
	v := reflect.ValueOf(q)
	for i := 0; i < v.NumField(); i++ {
		f := v.Type().Field(i)
		name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
		if !formSupported(f.Type) {
			t.Errorf("api.PredictJSONRequest.%s (%s): formInto cannot fill this kind from a multipart field", f.Name, f.Type)
		}
		fv := v.Field(i)
		if fv.IsZero() {
			t.Errorf("fullRequest leaves %q unset: give it a value so both content types are compared on it", name)
		}
		switch fv.Kind() {
		case reflect.String:
			out[name] = fv.String()
		case reflect.Int:
			out[name] = strconv.FormatInt(fv.Int(), 10)
		case reflect.Float64:
			out[name] = strconv.FormatFloat(fv.Float(), 'g', -1, 64)
		}
	}
	return out
}

// The tentpole of the request model: a multipart form and a JSON body carrying the same options
// must reach the model as the SAME prompt, and produce the same answer.
func TestEveryOptionMapsOneToOne(t *testing.T) {
	full, img, depth := fullRequest(t)
	jsonBody, _ := json.Marshal(full)

	fields := formOf(t, full)
	delete(fields, "image_base64") // multipart sends the binaries as file parts
	delete(fields, "depth_base64")
	mp := func() *http.Request {
		return multipartRequest(t, "/api/predict", fields, part{"image", "i.png", img}, part{"depth", "d.bin", depth})
	}

	run := func(req *http.Request) (models.Prompt, string, *httptest.ResponseRecorder) {
		f := &fakeRuntime{result: api.Result{Task: api.TaskDepth, Model: "m", DepthMap: []float32{1, 2}, DepthWidth: 2, DepthHeight: 1}}
		_, h := newTestServer(f)
		rec := do(h, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		return f.prompt, f.img.Bounds().String(), rec
	}
	pJSON, imgJSON, recJSON := run(jsonRequest("/api/predict", jsonBody))
	pForm, imgForm, recForm := run(mp())

	if !reflect.DeepEqual(pJSON, pForm) {
		t.Fatalf("JSON and multipart reach the model differently:\n json: %+v\n form: %+v", pJSON, pForm)
	}
	if imgJSON != imgForm || recJSON.Body.String() != recForm.Body.String() {
		t.Fatalf("different image (%s vs %s) or answer:\n%s\n%s", imgJSON, imgForm, recJSON.Body, recForm.Body)
	}

	// And the prompt is what each option means (spot-check the semantics, not just equality).
	p := pJSON
	if p.GridSize != maxGridSize {
		t.Errorf("grid_size 1000 not clamped: %d", p.GridSize)
	}
	// ROI 2,2,12,10 → the model sees a 12x10 crop, prompts shifted by (2,2), depth cropped too.
	if imgJSON != "(0,0)-(12,10)" || p.Boxes[0] != [4]float64{8, 6, 4, 2} || p.Points[0].X != 3 || p.Points[1].Label != 0 {
		t.Errorf("ROI crop/shift wrong: img %s boxes %v points %v", imgJSON, p.Boxes, p.Points)
	}
	if p.DepthW != 12 || p.DepthH != 10 || len(p.Depth) != 120 || math.IsNaN(float64(p.Depth[0])) {
		t.Errorf("depth not resized to the image then cropped to the ROI: %dx%d len %d", p.DepthW, p.DepthH, len(p.Depth))
	}
	if p.Text != "cat. remote." || p.MinSize != 0.5 || p.MaxSize != 90 || p.GripperMin != 3 || p.GripperMax != 40.5 ||
		p.BoxThresh != 0.35 || p.TextThresh != 0.2 || p.BgMaxArea != 50 || p.FgMinArea != 0.1 || p.Method != "sam" ||
		p.ClaimThresh != 0.6 || p.CropTemp != 0.05 || p.Dilate != -2 || p.TemplateName != "tpl" {
		t.Errorf("an option did not reach the prompt: %+v", p)
	}
	// encoding=base64 reached the response writer.
	if !strings.Contains(recJSON.Body.String(), `"depth_map_base64"`) {
		t.Errorf("encoding=base64 ignored: %s", recJSON.Body)
	}
}

// Multipart may also carry image_base64 / depth_base64 as text fields (the names are 1:1).
func TestMultipartAcceptsBase64Fields(t *testing.T) {
	full, _, _ := fullRequest(t)
	fields := formOf(t, full)
	f := &fakeRuntime{}
	_, h := newTestServer(f)
	if rec := do(h, multipartRequest(t, "/api/predict", fields)); rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if f.prompt.DepthW != 12 || f.img.Bounds().Dx() != 12 {
		t.Fatalf("base64 text fields not used: depth %d img %v", f.prompt.DepthW, f.img.Bounds())
	}
}

// The legacy multipart semantics: a malformed optional number is ignored (the option keeps its
// default), the encoding may come from the query string, and every field is optional but model
// and image.
func TestMultipartLegacySemantics(t *testing.T) {
	f := &fakeRuntime{result: api.Result{Embeddings: [][]float32{{1, 2}}}}
	_, h := newTestServer(f)
	req := multipartRequest(t, "/api/predict?encoding=base64",
		map[string]string{"model": "m", "min_size": "abc", "grid_size": "2.5", "dilate": "", "box_threshold": " 0.4"},
		part{"image", "i.png", pngBytes(t, 8, 8)})
	rec := do(h, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if p := f.prompt; p.MinSize != 0 || p.GridSize != 0 || p.Dilate != 0 || p.BoxThresh != 0 {
		t.Errorf("malformed numbers must leave defaults: %+v", p)
	}
	if !strings.Contains(rec.Body.String(), `"embeddings_base64"`) || !strings.Contains(rec.Body.String(), `"embeddings_shape":[1,2]`) {
		t.Errorf("?encoding=base64 ignored on multipart: %s", rec.Body)
	}
}

// JSON keeps json.Unmarshal's typing: a wrong type is a 400, as it always was.
func TestJSONTypeErrorIsBadRequest(t *testing.T) {
	_, h := newTestServer(&fakeRuntime{})
	rec := do(h, jsonRequest("/api/predict", []byte(`{"model":"m","image_base64":"x","grid_size":2.5}`)))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "invalid JSON body") {
		t.Fatalf("got %d %s", rec.Code, rec.Body)
	}
}

// The query parameter selects the encoding for a JSON body too; a bad name is a 400 before
// any work is done.
func TestEncodingFromQueryAndValidation(t *testing.T) {
	body, _ := json.Marshal(api.PredictJSONRequest{Model: "m", ImageBase64: b64(pngBytes(t, 4, 4))})
	f := &fakeRuntime{result: api.Result{DepthMap: []float32{0.5}, DepthWidth: 1, DepthHeight: 1}}
	_, h := newTestServer(f)
	if rec := do(h, jsonRequest("/api/predict?encoding=base64", body)); !strings.Contains(rec.Body.String(), "depth_map_base64") {
		t.Fatalf("?encoding=base64 ignored for JSON: %s", rec.Body)
	}
	f2 := &fakeRuntime{}
	_, h2 := newTestServer(f2)
	rec := do(h2, jsonRequest("/api/predict?encoding=msgpack", body))
	if rec.Code != http.StatusBadRequest || len(f2.Events()) != 0 {
		t.Fatalf("bad encoding: %d %s, events %v (want a 400 before admission)", rec.Code, rec.Body, f2.Events())
	}
}

// Missing pieces keep the exact messages each content type always answered.
func TestRequiredFieldMessages(t *testing.T) {
	_, h := newTestServer(&fakeRuntime{})
	cases := []struct {
		req  *http.Request
		want string
	}{
		{jsonRequest("/api/predict", []byte(`{"model":"m"}`)), "both 'model' and 'image_base64' are required"},
		{jsonRequest("/api/predict", []byte(`{"image_base64":"eA=="}`)), "both 'model' and 'image_base64' are required"},
		{multipartRequest(t, "/api/predict", map[string]string{"prompt": "x"}), "missing 'model' field"},
		{multipartRequest(t, "/api/predict", map[string]string{"model": "m"}), "missing 'image' file: http: no such file"},
		{multipartRequest(t, "/api/preprocess", map[string]string{"prompt": "x"}), "missing 'model' field"},
		{jsonRequest("/api/preprocess", []byte(`{"prompt":"x"}`)), "missing 'model' field"},
	}
	for _, c := range cases {
		rec := do(h, c.req)
		var e api.ErrorResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil || rec.Code != http.StatusBadRequest || e.Error != c.want {
			t.Errorf("%s: got %d %q, want 400 %q", c.req.URL, rec.Code, rec.Body, c.want)
		}
	}
}
