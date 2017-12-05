dims = csvread('dim.csv');
num_frames = dims(1);
height = dims(2);
width = dims(3);

F_csv = csvread('F.csv');
F = zeros(3, 3, num_frames-1);
for i=1:num_frames-1
    F(1,:,i) = F_csv(i,1:3);
    F(2,:,i) = F_csv(i,4:6);
    F(3,:,i) = F_csv(i,7:9);
end

crop_ratio = 0.8;
p_values = optimize_transforms(F, dims, crop_ratio);

p_csv = zeros(num_frames-1, 6);
for i=1:num_frames-1
    p_csv(i,:) = p_values(:,i);
end

csvwrite('p.csv', p_csv);

exit