# GPU_Image
Use gpu to generate images

# How to use:

edit function `calculate_pixel()` at `line: 166`<br>
`x` = pixel position x (`0` - `WIDTH-1`)<br>
`y` = pixel position y (`0` - `HEIGHT-1`)<br>
`index` = frame index  (`0` - `FRAME_COUNT-1`)<br>
`width` = `WIDTH`<br>
`height` = `HEIGHT`<br>
`colors[tid]` = assign as output color<br>
`uv` = pixel position mapped to `0`-`1` ranges

# Implemented functions:

`IS_ZERO` = checks if float is zero based on `EPSILON`<br>
`CROSS` = vector3 cross product<br>
`DOT` = vector3 dot product<br>
`DOT_2` = vector2 dot product<br>
`LENGTH` = vector3 length<br>
`LENGTH_2` = vector2 length<br>
`NORMALIZE` = normalized vector (both)<br>
`split_string` = split string using a seperator into `vector<string>`<br>
`sec_to_time` = convert `float` seconds to `string` second(s), minute(s), hour(s), day(s) 

# Constants:

`WIDTH` = image width<br>
`HEIGHT` = image height<br>
`FRAME_COUNT` = animation frame count<br>
`IS_ANIMATION` = toggle using animation<br>
`EPSILON` = arbitrary number (number close to zero)<br>
`M_PI` = PI constant to 8 digits<br>
