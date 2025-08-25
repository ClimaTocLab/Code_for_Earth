#!/bin/bash

source myenv/bin/activate

depths_list=("[5,5,5,5,5]" "[6,6,6,6,6]")
embed_dim_list=(64 128 256)
num_heads_list=("[5,5,5,5,5]" "[6,6,6,6,6]")

for depths in "${depths_list[@]}"; do
  for embed_dim in "${embed_dim_list[@]}"; do
    for num_heads in "${num_heads_list[@]}"; do
      echo "Running with depths=$depths, embed_dim=$embed_dim, num_heads=$num_heads"

      python3 train_cli.py fit \
        --config=configs/test.yaml \
        --overrides "model.depths=${depths}" "model.embed_dim=${embed_dim}" "model.num_heads=${num_heads}"

      dir = pwd
      echo $dir
      
      # Encuentra la primera carpeta que contiene "output" en su nombre
      output_dir=$(find . -type d -name "*output*" | head -n 1)
      
      # Encuentra el primer modelo que contiene "example" en su nombre
      ckpt_path=$(find "$output_dir/models" -type f -name "*example*" | head -n 1)
      
      echo "Model: $ckpt_path"
      
      # Verificación
      if [[ -z "$ckpt_path" ]]; then
          echo "❌ No se encontró ningún archivo que contenga 'example' en '$output_dir/models'"
          exit 1
      fi
      
      python3 train_cli.py validate \
          --config=configs/test_val.yaml \
          --ckpt_path="$ckpt_path"
      #    --overrides "trainer.strategy=ddp" \
      #    --overrides "data.batch_size=32" "data.num_workers=4"

      depths_clean=$(echo $depths | tr -d '[]' | tr ',' '-')
      num_heads_clean=$(echo $num_heads | tr -d '[]' | tr ',' '-')

      out_dir="AirPollutant_downscaling_${depths_clean}_${embed_dim}_${num_heads_clean}"
      echo "Output folder: $out_dir"

      # Crear carpeta de salida si no existe
      mkdir -p "$out_dir"

      # Mover carpetas si existen
      if [ -d "wandb" ]; then
        mv wandb "$out_dir/"
      fi

      if [ -d "tb_logs" ]; then
        mv tb_logs "$out_dir/"
      fi

      for d in output*; do
        if [ -d "$d" ]; then
          mv "$d" "$out_dir/"
        fi
      done

      echo "Finished run. Logs moved to $out_dir"
      echo "---------------------------"

    done
  done
done



