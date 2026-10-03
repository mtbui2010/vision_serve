package registry

import (
	"fmt"
	"reflect"
	"strings"

	"gopkg.in/yaml.v3"
)

// unknownManifestKeys lists the keys of a manifest's YAML that no Manifest field reads, as dotted
// paths with their line ("runtime.idle_unload_second (line 14)"). yaml.Unmarshal drops such keys
// silently, so a typo like `idle_unload_second:` would leave the real setting at its zero value.
//
// They are reported as a warning, not refused: a third-party manifest may legitimately carry
// extra keys (notes, keys of a newer VisionServe). The walk follows the yaml tags of Manifest;
// map values are walked with the map's element type, and a type with its own UnmarshalYAML
// (SHA256Field, wholeNumber) decides its own shape, so it is not walked into.
func unknownManifestKeys(raw []byte) ([]string, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	var out []string
	if doc.Kind == yaml.DocumentNode && len(doc.Content) > 0 {
		walkUnknownKeys(doc.Content[0], reflect.TypeOf(Manifest{}), "", &out)
	}
	return out, nil
}

var yamlUnmarshaler = reflect.TypeOf((*yaml.Unmarshaler)(nil)).Elem()

func walkUnknownKeys(n *yaml.Node, t reflect.Type, path string, out *[]string) {
	for n.Kind == yaml.AliasNode && n.Alias != nil {
		n = n.Alias
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if reflect.PointerTo(t).Implements(yamlUnmarshaler) {
		return
	}
	switch t.Kind() {
	case reflect.Struct:
		if n.Kind != yaml.MappingNode {
			return // a shape mismatch is yaml.Unmarshal's error to report, not ours
		}
		fields := yamlFields(t)
		for i := 0; i+1 < len(n.Content); i += 2 {
			k, v := n.Content[i], n.Content[i+1]
			if k.Value == "<<" { // merge key: its mapping(s) fill this same struct
				walkUnknownKeys(v, t, path, out)
				continue
			}
			ft, ok := fields[k.Value]
			if !ok {
				*out = append(*out, fmt.Sprintf("%s (line %d)", joinKeyPath(path, k.Value), k.Line))
				continue
			}
			walkUnknownKeys(v, ft, joinKeyPath(path, k.Value), out)
		}
	case reflect.Map:
		if n.Kind != yaml.MappingNode {
			return
		}
		for i := 0; i+1 < len(n.Content); i += 2 {
			walkUnknownKeys(n.Content[i+1], t.Elem(), joinKeyPath(path, n.Content[i].Value), out)
		}
	case reflect.Slice, reflect.Array:
		if n.Kind != yaml.SequenceNode {
			return
		}
		for i, item := range n.Content {
			walkUnknownKeys(item, t.Elem(), fmt.Sprintf("%s[%d]", path, i), out)
		}
	}
}

// yamlFields maps the YAML key of every field yaml.v3 decodes into t to the field's type, with
// the same rules as yaml.v3: the tag's name, else the lowercased field name; `-` and unexported
// fields are skipped; `,inline` structs contribute their own fields.
func yamlFields(t reflect.Type) map[string]reflect.Type {
	out := map[string]reflect.Type{}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if f.PkgPath != "" && !f.Anonymous {
			continue // unexported
		}
		tag := f.Tag.Get("yaml")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")
		if strings.Contains(","+opts+",", ",inline,") {
			ft := f.Type
			for ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				for k, v := range yamlFields(ft) {
					out[k] = v
				}
			}
			continue
		}
		if name == "" {
			name = strings.ToLower(f.Name)
		}
		out[name] = f.Type
	}
	return out
}

func joinKeyPath(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}
