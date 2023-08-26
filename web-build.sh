rm -rf public
mkdir -p public/{pkg,res}

cp index.html public/
cp -R res/ public/res/
wasm-pack build --target web --release --out-dir public/pkg
rm public/pkg/.gitignore
