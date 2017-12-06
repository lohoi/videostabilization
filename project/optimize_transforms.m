function p_values = optimize_transforms(F, vid_shape, crop_ratio)
% Numebr of transformations required
f_len = vid_shape(1) - 1;
% Weighting on Affine Components from Paper
affine_weighting = [1 1 100 100 100 100];
% Weights on different parts of path smoothing
w1 = 10;
w2 = 1;
w3 = 100;
c = [w1 w2 w3];

% Begin LP
cvx_begin

variable p(6,f_len)
variable e1(6,f_len)
variable e2(6,f_len)
variable e3(6,f_len)

minimize(sum(c(1) * affine_weighting * e1 + c(2) * affine_weighting * e2 + c(3) * affine_weighting * e3));
subject to
    % Smoothness Constraints
    for i=1:f_len - 3
        B_t = p_to_b_mat(p(:,i));
        B_t1 = p_to_b_mat(p(:,i+1));
        B_t2 = p_to_b_mat(p(:,i+2));
        B_t3 = p_to_b_mat(p(:,i+3));

        R_t = F(:, :, i + 1) * B_t1 - B_t;
        R_t1 = F(:, :, i + 2) * B_t2 - B_t1;
        R_t2 = F(:, :, i + 3) * B_t3 - B_t2;

        R_t = mat_to_col(R_t);
        R_t1 = mat_to_col(R_t1);
        R_t2 = mat_to_col(R_t2);
        % Constraints on e1,e2,e3, and p as stated in the paper (algorithm 1)
        -e1(:,i) <= R_t <= e1(:,i);
        -e2(:,i) <= R_t1 - R_t <= e2(:,i);
        -e3(:,i) <= R_t2 - 2 * R_t1 + R_t <= e3(:,i);
    end
    %%% Positive Slack Variable Constraints %%% 
    for i=1:f_len
        e1(:,i) >= 0;
        e2(:,i) >= 0;
        e3(:,i) >= 0;
    end
    for i=f_len-3:f_len
        p(:,i) == p(:,f_len)
    end
    
    % Proximity Constraints
    lb = [0.9 0.9 -0.1 -0.1 -0.05 -0.1]';
    ub = [1.1 1.1 0.1 0.1 0.05 0.1]';
    U = [0, 0, 1, 0, 0, 0; 0, 0, 0, 0, 0, 1; 0, 0, 0, 1, 0, 0; 0, 0, 0, 0, 1, 0; 0, 0, 0, 1, 1, 0; 0, 0, 0, 1, -1, 0];
    for i=1:f_len
        lb <= U*p(:,i) <= ub;
    end
    % Inclusion Constraints
    center = [vid_shape(3)/2 vid_shape(2)/2];
    crop_width = crop_ratio * vid_shape(3);
    crop_height = crop_ratio * vid_shape(2);
    corners = [center(1)-crop_width/2 center(2)-crop_height/2; center(1)+crop_width/2 center(2)-crop_height/2; center(1)-crop_width/2 center(2)+crop_height/2; center(1)+crop_width/2 center(2)+crop_height/2]; 
    for i=1:f_len
        for j=1:4
            0 <= [1 0 corners(j,1) corners(j,2) 0 0] * p(:,i) <= vid_shape(3);
            0 <= [0 1 0 0 corners(j,1) corners(j,2)] * p(:,i) <= vid_shape(2);
        end
    end
cvx_end
% end LP
p_values = p;
for i=1:f_len
    p_values(:,i) = p2p(p_values(:,i));
end

end

function B = p_to_b_mat(p)
    B = [p(3) p(4) p(1); p(5) p(6) p(2); 0 0 1];
end

function p = mat_to_col(R)
    p = [R(1,3) R(2,3) R(1,1) R(1,2) R(2,1) R(2,2)]';
end

function p = p2p(p)
    p = [p(3) p(4) p(1) p(5) p(6) p(2)]';
end

