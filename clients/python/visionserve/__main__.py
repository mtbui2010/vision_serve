from visionserve.cli import main

# Exit with main()'s status, like the `visionserve` console script does: `python -m visionserve`
# used to exit 0 even after printing an error.
raise SystemExit(main())
