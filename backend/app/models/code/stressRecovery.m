function [ stress_gp_cell, stress_node_cell,strain_gp_cell,strain_node_cell] = ...
    stressRecovery( displacements, numberElements,elementNodes,...
    nodeCoordinates,D,thickness,materialMap,stiffnessReduction)
%STRESSRECOVERY Recover element stresses from Gauss points


[gaussWeights,gaussLocations]=gauss2d('2x2');
stress_gp_cell=cell(numberElements,1);
stress_node_cell=cell(numberElements,1);
strain_gp_cell=cell(numberElements,1);
strain_node_cell=cell(numberElements,1);

for e=1:numberElements
    numNodePerElement = length(elementNodes(e,:));
    numEDOF = 2*numNodePerElement;
    elementDof=zeros(1,numEDOF);
    for i = 1:numNodePerElement
        elementDof(2*i-1)=2*elementNodes(e,i)-1;
        elementDof(2*i)=2*elementNodes(e,i);
    end
    stress_gp=zeros(size(gaussWeights,1),3);
    strain_gp=zeros(size(gaussWeights,1),3);
    % cycle for Gauss point
    for q=1:size(gaussWeights,1)
        GaussPoint=gaussLocations(q,:);
        xi=GaussPoint(1);
        eta=GaussPoint(2);
        
        % shape functions and derivatives
        [~,naturalDerivatives]=shapeFunctionQ4(xi,eta);
        
        % Jacobian matrix, inverse of Jacobian,
        % derivatives w.r.t. x,y
        [~,~,XYderivatives]=...
            Jacobian(nodeCoordinates(elementNodes(e,:),:),naturalDerivatives);
        
        %  B matrix
        B=zeros(3,numEDOF);
        B(1,1:2:numEDOF)       = XYderivatives(1,:);
        B(2,2:2:numEDOF)  = XYderivatives(2,:);
        B(3,1:2:numEDOF)       = XYderivatives(2,:);
        B(3,2:2:numEDOF)  = XYderivatives(1,:);
        
        stiffnessReductionMAT=(1-(1-materialMap(e))*(1-stiffnessReduction))*D;
        
        % stress at gauss point
        strain_gp(q,:) = (B*displacements(elementDof))';
        stress_gp(q,:) = (stiffnessReductionMAT*thickness*B*displacements(elementDof))';
        
    end
    recovMat=[1+0.5*sqrt(3) -0.5 1-0.5*sqrt(3) -0.5;
        -0.5 1+0.5*sqrt(3) -0.5 1-0.5*sqrt(3);
        1-0.5*sqrt(3) -0.5 1+0.5*sqrt(3) -0.5;
        -0.5 1-0.5*sqrt(3) -0.5 1+0.5*sqrt(3);];
    stress_node=recovMat*stress_gp;
    strain_node=stress_node/stiffnessReductionMAT;
    strain_gp_cell{e}=strain_gp;
    stress_gp_cell{e}=stress_gp;
    strain_node_cell{e}=strain_node;
    stress_node_cell{e}=stress_node;
end

