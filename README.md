This fork aims to fix various mesh issues as of v0.66 from the main repository.<br>
**What's mainly changed?**<br>
1. Custom weights are now more accurate. There is a hard limitation for the precision of weights that can be stored in the mesh format. However, with this they are handled a bit differently to get closer to the original weight. Note, that I intentionally used the words 'custom weights', as vanilla mesh weights should already be packed in a format that keeps weights 1:1 between import/export.<br>
Cube test example with custom weights for each group:<br>
Before Export:<br>
0.165, 0.207, 0.092, 0.194, 0.191, 0.150<br>
After Export (v0.66, original)<br>
0.165, 0.204, 0.094, 0.196, 0.192, 0.149 Total precision loss: 0.009<br>
After Export (v0.67, forked meshfix)<br>
0.165, 0.208, 0.090, 0.196, 0.192, 0.149 Total precision loss: 0.007<br>
(Note; for this example I just used blender's display of to the nearst thousandth rather than the entire float)<br>

2. Exported normals (and other various data) can be fully preserved<br>
The option 'Split Vertices For Corner Attributes' has been added for this. You can disable to use the 'legacy' behavior.
<img width="2322" height="1259" alt="Desktop Screenshot 2026 06 10 - 18 06 32 582" src="https://github.com/user-attachments/assets/1a7b4e5e-c8c7-4009-a2cc-cf800cdd489d" />

3. Exported Weights Have the Option to Not be Forced Into Normalization<br>
The checkbox for export 'Normalize Weights' has been added. The reason for this is because some vanilla meshes actually are not fully normalized. So their weight was not able to be preserved 1:1 after export (or if you are trying to export something that matches the weight). Example being Grace from RE9's head mesh. There are weights around the neck seam near the shoulder area that are not fully normalized, and could not be exported with those weights unchanged.

4. Fixed DD2 Secondary Weight Issue<br>
<img width="2838" height="1445" alt="DD2" src="https://github.com/user-attachments/assets/9004eced-9f12-4b41-b0db-fe7037be26e2" />

5. Support for MHWs blendshapes

6. Support for Onimusha: WoTS meshes

7. Added new user configuration and export settings

8. Sped up some functions such as bone index packing

9. Extended weights packed closer to vanilla structure (only Wilds + Oni:WoTS at the time of writing this)


##
For additional help, go here:
[Modding World Discord Server](https://discord.gg/yA969DHdxM)
## Credits Specific to this fork
[HeartBee.](https://github.com/StellarBladeModding) - For helping test extended weight packing.


########### Original Repository ###################
[RE-Mesh-Editor](https://github.com/NSACloud/RE-Mesh-Editor)